import os.path as osp
import torch
import openai
import time
import asyncio
import numpy as np
from tqdm import tqdm
import pylcs
import html
import replicate
from recbole.model.abstract_recommender import SequentialRecommender

from utils import dispatch_openai_requests, dispatch_single_openai_requests


class Rank(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.max_tokens = config['max_tokens']
        self.api_model_name = config['api_name']
        openai.api_key = config['api_key']
        openai.api_base = config['api_base']
        self.api_batch = config['api_batch']
        self.async_dispatch = config['async_dispatch']
        self.temperature = config['temperature']

        self.max_his_len = config['max_his_len']
        self.recall_budget = config['recall_budget']
        self.boots = config['boots']
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        self.id_token = dataset.field2id_token['item_id']
        self.item_text = self.load_text()
        self.logger.info(f'Avg. t = {np.mean([len(_) for _ in self.item_text])}')

        self.fake_fn = torch.nn.Linear(1, 1)

    def load_text(self):
        token_text = {}
        token_preferences = {}  # Store preference tags separately
        item_text = ['[PAD]']
        self.item_preferences = {}  # item_index -> list of preference tags

        feat_path = osp.join(self.data_path, f'{self.dataset_name}.item')
        if self.dataset_name in ['ml-1m', 'ml-1m-full']:
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, movie_title, release_year, genre = line.strip().split('\t')
                    token_text[item_id] = movie_title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                if raw_text.endswith(', The'):
                    raw_text = 'The ' + raw_text[:-5]
                elif raw_text.endswith(', A'):
                    raw_text = 'A ' + raw_text[:-3]
                item_text.append(raw_text)
            return item_text
        elif self.dataset_name in ['Games', 'Games-6k']:
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, title = line.strip().split('\t')
                    token_text[item_id] = title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text
        elif self.dataset_name.startswith(('food_', 'movie_', 'amazon_', 'yelp_')):
            # Universal handler for food, movie, amazon, yelp with any tag variant
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        item_id, item_name, tags = parts[0], parts[1], parts[2]
                        tag_list = tags.split()

                        # Separate preference tags from native tags
                        preference_tags = []
                        native_tags = []
                        for tag in tag_list:
                            # Check for both English and Chinese preference suffixes
                            if ' Preference' in tag or '偏好' in tag:
                                # Remove the suffix to get the clean preference name
                                clean_tag = tag.replace(' Preference', '').replace('偏好', '')
                                preference_tags.append(clean_tag)
                            else:
                                native_tags.append(tag)

                        # Store native tags with item name
                        if native_tags:
                            token_text[item_id] = f"{item_name} ({' '.join(native_tags)})"
                        else:
                            token_text[item_id] = item_name

                        # Store preference tags separately
                        if preference_tags:
                            token_preferences[item_id] = preference_tags

                    elif len(parts) == 2:
                        item_id, item_name = parts
                        token_text[item_id] = item_name

            # Build item_text list and map preferences to indices
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text.get(token, f'Item_{token}')
                item_text.append(raw_text)

                # Map preference tags to item index
                if token in token_preferences:
                    self.item_preferences[len(item_text) - 1] = token_preferences[token]

            return item_text
        else:
            raise NotImplementedError()

    def predict_on_subsets(self, interaction, idxs):
        """
        Main function to rank with LLMs

        :param interaction:
        :param idxs: item id retrieved by candidate generation models [batch_size, candidate_size]
        :return:
        """
        origin_batch_size = idxs.shape[0]
        if self.boots:
            """ 
            bootstrapping is adopted to alleviate position bias
            `fix_enc` is invalid in this case"""
            idxs = np.tile(idxs, [self.boots, 1])
            np.random.shuffle(idxs.T)
        batch_size = idxs.shape[0]
        pos_items = interaction[self.POS_ITEM_ID]

        prompt_list = []
        for i in tqdm(range(batch_size)):
            user_his_text, candidate_text, candidate_text_order, candidate_idx, session_pref_counts, candidate_prefs = self.get_batch_inputs(interaction, idxs, i)

            prompt = self.construct_prompt(self.dataset_name, user_his_text, candidate_text_order, session_pref_counts, candidate_prefs)
            prompt_list.append([{'role': 'user', 'content': prompt}])

        if 'llama' in self.api_model_name:
            openai_responses = self.dispatch_replicate_api_requests(prompt_list, batch_size)
        else:
            openai_responses = self.dispatch_openai_api_requests(prompt_list, batch_size)

        scores = torch.full((idxs.shape[0], self.n_items), -10000.)
        for i, openai_response in enumerate(tqdm(openai_responses)):
            retry_flag = 1
            while retry_flag >= 0:
                user_his_text, candidate_text, candidate_text_order, candidate_idx, session_pref_counts, candidate_prefs = self.get_batch_inputs(interaction, idxs, i)

                if 'llama' in self.api_model_name:
                    response = openai_response
                else:
                    response = openai_response['choices'][0]['message']['content']
                response_list = response.split('\n')
                
                self.logger.info(prompt_list[i])
                self.logger.info(response)
                self.logger.info(f'Here are candidates: {candidate_text}')
                self.logger.info(f'Here are answer: {response_list}')
                
                if self.dataset_name in ['ml-1m', 'ml-1m-full']:
                    rec_item_idx_list = self.parsing_output_text(scores, i, response_list, idxs, candidate_text)
                elif self.dataset_name in ['Games', 'Games-6k']:
                    # rec_item_idx_list = self.parsing_output_indices(scores, i, response_list, idxs, candidate_text)
                    rec_item_idx_list = self.parsing_output_text(scores, i, response_list, idxs, candidate_text)
                elif self.dataset_name.startswith(('food_', 'movie_', 'amazon_', 'yelp_')):
                    # Use text parsing for food, movie, amazon, yelp datasets
                    rec_item_idx_list = self.parsing_output_text(scores, i, response_list, idxs, candidate_text)
                else:
                    raise NotImplementedError(f"Dataset {self.dataset_name} not supported")

                if int(pos_items[i % origin_batch_size]) in candidate_idx:
                    target_text = candidate_text[candidate_idx.index(int(pos_items[i % origin_batch_size]))]
                    try:
                        ground_truth_pr = rec_item_idx_list.index(target_text)
                        self.logger.info(f'Ground-truth [{target_text}]: Ranks {ground_truth_pr}')
                        retry_flag = -1
                    except:
                        if 'llama' in self.api_model_name:
                            retry_flag = -1
                        else:
                            self.logger.info(f'Fail to find ground-truth items.')
                            print(target_text)
                            print(rec_item_idx_list)
                            print(f'Remaining {retry_flag} times to retry.')
                            retry_flag -= 1
                            while True:
                                try:
                                    openai_response = dispatch_single_openai_requests(prompt_list[i], self.api_model_name, self.temperature)
                                    break
                                except Exception as e:
                                    print(f'Error {e}, retry at {time.ctime()}', flush=True)
                                    time.sleep(20)
                else:
                    retry_flag = -1

        if self.boots:
            scores = scores.view(self.boots,-1,scores.size(-1))
            scores = scores.sum(0)
        return scores

    def get_batch_inputs(self, interaction, idxs, i):
        from collections import Counter

        user_his = interaction[self.ITEM_SEQ]
        user_his_len = interaction[self.ITEM_SEQ_LEN]
        origin_batch_size = user_his.size(0)
        real_his_len = min(self.max_his_len, user_his_len[i % origin_batch_size].item())
        user_his_text = [str(j) + '. ' + self.item_text[user_his[i % origin_batch_size, user_his_len[i % origin_batch_size].item() - real_his_len + j].item()] \
                for j in range(real_his_len)]
        candidate_text = [self.item_text[idxs[i,j]]
                for j in range(idxs.shape[1])]
        candidate_text_order = [str(j) + '. ' + self.item_text[idxs[i,j].item()]
                for j in range(idxs.shape[1])]
        candidate_idx = idxs[i].tolist()

        # Extract preference tags from history and count session-level frequencies
        session_pref_counts = Counter()
        for j in range(real_his_len):
            item_idx = user_his[i % origin_batch_size, user_his_len[i % origin_batch_size].item() - real_his_len + j].item()
            if item_idx in self.item_preferences:
                for pref in self.item_preferences[item_idx]:
                    session_pref_counts[pref] += 1

        # Extract preference tags from candidates
        candidate_prefs = []
        for j in range(idxs.shape[1]):
            item_idx = idxs[i,j].item()
            if item_idx in self.item_preferences:
                candidate_prefs.append(self.item_preferences[item_idx])
            else:
                candidate_prefs.append([])

        return user_his_text, candidate_text, candidate_text_order, candidate_idx, session_pref_counts, candidate_prefs

    def construct_prompt(self, dataset_name, user_his_text, candidate_text_order, session_pref_counts=None, candidate_prefs=None):
        # Build session preference summary if available
        pref_section = ""
        if session_pref_counts and len(session_pref_counts) > 0:
            pref_summary = "\n".join([f"  - {pref}: {count} items" for pref, count in session_pref_counts.most_common()])

        # Build candidate list with preference alignment if available
        candidate_display = candidate_text_order
        if candidate_prefs and session_pref_counts and len(session_pref_counts) > 0:
            # Reconstruct candidate list with preference information
            candidate_lines = candidate_text_order.split('\n') if isinstance(candidate_text_order, str) else candidate_text_order
            enhanced_candidates = []
            for idx, (cand_text, cand_prefs) in enumerate(zip(candidate_lines, candidate_prefs)):
                if cand_prefs:
                    # Show which preferences match session
                    matching = [p for p in cand_prefs if p in session_pref_counts]
                    if matching:
                        pref_str = ", ".join(matching)
                        enhanced_candidates.append(f"{cand_text}\n   [Matching visitor patterns: {pref_str}]")
                    else:
                        enhanced_candidates.append(cand_text)
                else:
                    enhanced_candidates.append(cand_text)
            candidate_display = '\n'.join(enhanced_candidates)
        elif isinstance(candidate_text_order, list):
            candidate_display = '\n'.join(candidate_text_order)

        if dataset_name in ['ml-1m', 'ml-1m-full']:
            prompt = f"I've watched the following movies in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_display}\n" \
                    f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
        elif dataset_name in ['Games', 'Games-6k']:
            prompt = f"I've purchased the following products in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate products that I can consider to purchase next:\n{candidate_display}\n" \
                    f"Please rank these {self.recall_budget} products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
                    # f"Please only output the order numbers after ranking. Split these order numbers with line break."
        elif dataset_name.startswith('food_'):
            # Food dataset (Chinese)
            if session_pref_counts and len(session_pref_counts) > 0:
                pref_summary = "\n".join([f"  - {pref}: {count} 項商品" for pref, count in session_pref_counts.most_common()])
                pref_section = f"\n基於您的購買歷史,這個會話顯示出以下顧客偏好模式:\n{pref_summary}\n"
            prompt = f"我過去按照順序購買了以下食品:\n{user_his_text}\n{pref_section}\n" \
                    f"現在有 {self.recall_budget} 個候選食品可以考慮接下來購買:\n{candidate_display}\n" \
                    f"請根據我的購買歷史和偏好模式,對這 {self.recall_budget} 個食品進行排序,衡量我最有可能接下來購買的食品。請逐步思考。\n" \
                    f"請使用順序編號顯示您的排序結果。用換行符分隔輸出。您必須對給定的候選食品進行排序。您不能生成不在給定候選列表中的食品。"
        elif dataset_name.startswith('movie_'):
            # Movie dataset (English)
            if session_pref_counts and len(session_pref_counts) > 0:
                pref_section = f"\nBased on your viewing history, this session shows the following viewer preference patterns:\n{pref_summary}\n"
            prompt = f"I've watched the following movies in the past in order:\n{user_his_text}\n{pref_section}\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_display}\n" \
                    f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history and preference patterns. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
        elif dataset_name.startswith('amazon_'):
            # Amazon dataset (English)
            if session_pref_counts and len(session_pref_counts) > 0:
                pref_section = f"\nBased on your purchase history, this session shows the following shopper preference patterns:\n{pref_summary}\n"
            prompt = f"I've purchased the following products in the past in order:\n{user_his_text}\n{pref_section}\n" \
                    f"Now there are {self.recall_budget} candidate products that I can consider to purchase next:\n{candidate_display}\n" \
                    f"Please rank these {self.recall_budget} products by measuring the possibilities that I would like to purchase next most, according to my purchase history and preference patterns. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate products. You can not generate products that are not in the given candidate list."
        elif dataset_name.startswith('yelp_'):
            # Yelp dataset (English) - restaurants/businesses
            if session_pref_counts and len(session_pref_counts) > 0:
                pref_section = f"\nBased on your visit history, this session shows the following visitor preference patterns:\n{pref_summary}\n"
            prompt = f"I've visited the following businesses in the past in order:\n{user_his_text}\n{pref_section}\n" \
                    f"Now there are {self.recall_budget} candidate businesses that I can consider to visit next:\n{candidate_display}\n" \
                    f"Please rank these {self.recall_budget} businesses by measuring the possibilities that I would like to visit next most, according to my visit history and preference patterns. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate businesses. You can not generate businesses that are not in the given candidate list."
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt

    def dispatch_openai_api_requests(self, prompt_list, batch_size):
        openai_responses = []
        self.logger.info('Launch OpenAI APIs')
        if self.async_dispatch:
            self.logger.info('Asynchronous dispatching OpenAI API requests.')
            for i in tqdm(range(0, batch_size, self.api_batch)):
                max_retries = 3
                retry_count = 0
                batch_num = i // self.api_batch
                while retry_count < max_retries:
                    try:
                        batch_responses = asyncio.run(
                            dispatch_openai_requests(prompt_list[i:i+self.api_batch], self.api_model_name, self.temperature)
                        )
                        openai_responses += batch_responses
                        print(f'✓ Batch {batch_num}/{(batch_size-1)//self.api_batch} completed ({len(batch_responses)} responses)', flush=True)
                        self.logger.info(f'Batch {batch_num} completed successfully with {len(batch_responses)} responses')
                        # Log sample response - show the ranking part (last 300 chars which contains the numbered list)
                        if batch_responses:
                            full_content = batch_responses[0]['choices'][0]['message']['content'] if 'choices' in batch_responses[0] else 'N/A'
                            # Extract just the ranking portion (lines starting with numbers)
                            lines = full_content.split('\n')
                            ranking_lines = [line for line in lines if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('0.'))]
                            if ranking_lines:
                                ranking_text = '\n'.join(ranking_lines[:10])  # Show first 10 ranked items
                                self.logger.info(f'Sample ranking from batch {batch_num}:\n{ranking_text}')
                            else:
                                # Fallback: show last 300 chars
                                self.logger.info(f'Sample response from batch {batch_num} (last 300 chars): ...{full_content[-300:]}')
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f'✗ Max retries ({max_retries}) reached for batch {batch_num}, adding fallback responses...', flush=True)
                            self.logger.warning(f'Batch {batch_num} failed after {max_retries} retries, using fallback responses')
                            # Add dummy responses to maintain batch alignment
                            num_failed = min(self.api_batch, batch_size - i)
                            for _ in range(num_failed):
                                fallback_response = {
                                    'choices': [{'message': {'content': '0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\n18\n19'}}],
                                    'error': 'Max retries exceeded'
                                }
                                openai_responses.append(fallback_response)
                            self.logger.warning(f'Added {num_failed} fallback responses for batch {batch_num}')
                            break
                        print(f'⚠ Error {e}, retry {retry_count}/{max_retries} for batch {batch_num} at {time.ctime()}', flush=True)
                        time.sleep(20)
                # Small delay between batches to avoid rate limiting
                time.sleep(0.1)
        else:
            self.logger.info('Dispatching OpenAI API requests one by one.')
            for message in tqdm(prompt_list):
                openai_responses.append(dispatch_single_openai_requests(message, self.api_model_name, self.temperature))
        self.logger.info('Received OpenAI Responses')
        return openai_responses
    
    def dispatch_replicate_api_requests(self, prompt_list, batch_size):
        responses = []
        self.logger.info('Launch Replicate APIs')
        suffix = {
            'llama-2-7b-chat': '4b0970478e6123a0437561282904683f32a9ed0307205dc5db2b5609d6a2ceff',
            'llama-2-70b-chat': '2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1'
        }[self.api_model_name]
        for message in tqdm(prompt_list):
            while True:
                try:
                    output = replicate.run(
                        f"meta/{self.api_model_name}:{suffix}",
                        input={"prompt": f"[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n{message[0]['content']}[/INST]"}
                    )
                    break
                except Exception as e:
                    print(f'Error {e}, retry at {time.ctime()}', flush=True)
                    time.sleep(20)

            responses.append(''.join([_ for _ in output]))
        return responses

    def parsing_output_text(self, scores, i, response_list, idxs, candidate_text):
        rec_item_idx_list = []
        found_item_cnt = 0
        for j, item_detail in enumerate(response_list):
            if len(item_detail) < 1:
                continue
            if item_detail.endswith('candidate movies:'):
                continue
            pr = item_detail.find('. ')
            if item_detail[:pr].isdigit():
                item_name = item_detail[pr + 2:]
            else:
                item_name = item_detail

            matched_name = None
            for candidate_text_single in candidate_text:
                clean_candidate_text_single = html.unescape(candidate_text_single.strip())
                if (clean_candidate_text_single in item_name) or (item_name in clean_candidate_text_single) or (pylcs.lcs_sequence_length(item_name, clean_candidate_text_single) > 0.9 * len(clean_candidate_text_single)):
                    if candidate_text_single in rec_item_idx_list:
                        break
                    rec_item_idx_list.append(candidate_text_single)
                    matched_name = candidate_text_single
                    break
            if matched_name is None:
                continue

            candidate_pr = candidate_text.index(matched_name)
            scores[i, idxs[i, candidate_pr]] = self.recall_budget - found_item_cnt
            found_item_cnt += 1
        return rec_item_idx_list

    def parsing_output_indices(self, scores, i, response_list, idxs, candidate_text):
        rec_item_idx_list = []
        found_item_cnt = 0
        for j, item_detail in enumerate(response_list):
            if len(item_detail) < 1:
                continue

            if not item_detail.isdigit():
                continue

            pr = int(item_detail)
            if pr >= self.recall_budget:
                continue
            matched_name = candidate_text[pr]
            if matched_name in rec_item_idx_list:
                continue
            rec_item_idx_list.append(matched_name)
            scores[i, idxs[i, pr]] = self.recall_budget - found_item_cnt
            found_item_cnt += 1
            if len(rec_item_idx_list) >= self.recall_budget:
                break

        return rec_item_idx_list
