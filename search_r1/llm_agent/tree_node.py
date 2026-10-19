from __future__ import annotations

import math
from typing import List, Optional, Callable
from pydantic import BaseModel
import json
import random
from collections import deque
import torch
from verl import DataProto
import random
import os
import re
import string

DEBUG: bool = os.environ.get('TREE_SEARCH_DEBUG', '').lower() == 'true'

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class TreeNode:
    def __init__(
        self,
        tree_uid: str, # equal to the original question prompt uid
        node_uid: str,
        prompts: Optional[torch.Tensor] = None, # prompts are same as the original input_ids
        input_ids: Optional[torch.Tensor] = None, # input_ids are the left_part input during rollout
        attention_mask: Optional[torch.Tensor] = None, # attention_mask are the left_part input during rollout
        position_ids: Optional[torch.Tensor] = None, # position_ids are the left_part input during rollout
        responses: Optional[torch.Tensor] = None, # responses are the right_part output during rollout
        responses_with_info_mask: Optional[torch.Tensor] = None, # responses_with_info_mask are the right_part output during rollout
        turns_mask: Optional[torch.Tensor] = None, # turn_mask are the right_part output during rollout
        log_prob_node: Optional[float] = 0.0,
        log_prob_list: Optional[list[float]] = [],
        parent_node: Optional['TreeNode'] = None,
        is_root: bool = False,
        is_active: bool = True,
        valid_action_stats: int = 0,
        valid_search_stats: int = 0,
        depth: int = 0,
        is_leaf: bool = False,
        correct_leaf_in_subtree: int = 0,
        reward_mode: str = 'base', 
        tensor_fn = None,
        margin = 0.1,
    ):

        self.tree_uid: int = tree_uid
        self.node_uid: int = node_uid

        self.prompts: torch.Tensor = prompts
        self.input_ids: torch.Tensor = input_ids
        self.attention_mask: torch.Tensor = attention_mask
        self.position_ids: torch.Tensor = position_ids
        self.responses: torch.Tensor = responses
        self.responses_with_info_mask: torch.Tensor = responses_with_info_mask
        self.turns_mask: torch.Tensor = turns_mask

        self.log_prob_node: float = log_prob_node
        self.log_prob_list: list[float] = log_prob_list

        self.parent_node = parent_node
        self._child_node = []

        self.is_root = is_root
        self.is_active = is_active
        self.valid_action_stats = valid_action_stats
        self.valid_search_stats = valid_search_stats
        self.depth = depth
        self.is_leaf = is_leaf

        self.original_score = 0.
        self.final_score = 0.

        self.subtree_leaf_score = 0.

        self.reward_mode = reward_mode
        self.margin = margin

        self.tensor_fn = tensor_fn

    @property
    def child_node(self) -> list['TreeNode']:
        return self._child_node

    @child_node.setter
    def child_node(self, value: list['TreeNode']):
        """
        Debug
        """
        print(f"!!! WARNING: Direct assignment to child_node on node {self.node_uid}. New list has {len(value)} children.")
        
        import traceback
        traceback.print_stack()
        
        # Check
        for child in value:
            if child is self:
                raise ValueError(f"CRITICAL: Attempted to directly assign a list containing the node itself as its own child. Node UID: {self.node_uid}")

        self._child_node = value
        
    def add_child(self, child_node: 'TreeNode'):
        if child_node is self:
            raise ValueError("A node cannot be its own child")
        if child_node.node_uid == self.node_uid:
            raise ValueError("node_uid same!! A node cannot be its own child")
        self._child_node.append(child_node)
    
    def get_subtree_nodes(self):
        """
        Dynamically get all descendant nodes by traversing the tree
        """
        nodes = []
        nodes_to_visit = list(self.child_node) # from child nodes
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            nodes.append(current_node)
            nodes_to_visit.extend(current_node.child_node)
        return nodes

    def get_subtree_leaves_num(self):
        num = 0
        nodes_to_visit = list(self.child_node)
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            if current_node.is_leaf:
                num += 1
            nodes_to_visit.extend(current_node.child_node)
        return num

    def get_expand_node(self, n: int = 1, mode: str = 'random') -> List['TreeNode']:
        """
        Sample n nodes from the subtree
        """
        candidate_set = [self]
        for node in self.get_subtree_nodes():
            if not node.is_leaf:
                candidate_set.append(node)
        result = random.choices(candidate_set, k=n)
                
        assert len(result) == n, f"get_expand_node error, len(result)={len(result)} != n={n}"
        return result

    def sample_leaf(self, n: int = 1) -> List['TreeNode']:
        """
        Sample n leaves, then prune the tree (drop the unselected nodes)
        """

        candidate_uid_set = []

        for node in self.get_subtree_nodes():
            if node.is_leaf:
                candidate_uid_set.append(node.node_uid)

        if len(candidate_uid_set) < n:
            dprint(f"root={self.node_uid}, candidate_uid_set={candidate_uid_set}")
            subtree_nodes = self.get_subtree_nodes()
            subtree_node_uids = [node.node_uid for node in subtree_nodes]
            dprint(f"all subtree nodes={subtree_node_uids}")
        assert len(candidate_uid_set) >= n, f"root={self.node_uid}, candidate_uid_set len={len(candidate_uid_set)} < n={n}"

        random.shuffle(candidate_uid_set)
        dprint(f'original candidate_uid_set len={len(candidate_uid_set)}')
        candidate_uid_set = candidate_uid_set[:n]
        dprint(f'sampled candidate_uid_set len={len(candidate_uid_set)}')
        self._prune_subtree(candidate_uid_set)

        result = []
        for node in self.get_subtree_nodes():
            if node.is_leaf:
                result.append(node)
        return result

    def set_leaf_original_score(self, score: float):
        """
        Set the original score of the leaf
        """
        self.original_score = score

    @staticmethod
    def dfs_subtree_leaf_score(tmp_node: 'TreeNode') -> float:
        """
        Do dfs and compute the subtree leaf original score
        """
        subtree_leaf_score = tmp_node.original_score
        for node in tmp_node.child_node:
            subtree_leaf_score += TreeNode.dfs_subtree_leaf_score(node)
        tmp_node.subtree_leaf_score = subtree_leaf_score
        return subtree_leaf_score

    def calculate_final_score_from_root(self):
        """
        Calculate the Diff-based final score from root (not for Tree-GRPO. Tree-GRPO uses base mode)
        """

        # First do dfs and compute the subtree leaf original score
        TreeNode.dfs_subtree_leaf_score(self)

        # Then compute the final score for each node
        total_leaf_num = self.get_subtree_leaves_num()
        global_score_mean = self.subtree_leaf_score / total_leaf_num
        for node in self.get_subtree_nodes():
            # TreeRL global score
            curr_leaf_num = 1 if node.is_leaf else node.get_subtree_leaves_num()
            subtree_nodes = node.get_subtree_nodes()
            subtree_node_uids = [x.node_uid for x in subtree_nodes]
            assert curr_leaf_num > 0, f"node_uid={node.node_uid} have no leaves, subtree_node_uids={subtree_node_uids}"
            curr_score_mean = node.subtree_leaf_score / curr_leaf_num
            # ====================
            # This Part should be modified as the score within the subtrees cannot be compared. They come from different query intent. Should not be compared
            # This part should use a single-step decision score where the credit is assigned independently. The better single step should not be penalized.
            # But How to compare the advantages between the bad results with good steps and the good results with good/bad steps.
            # How to preserve the completeness of the rollout sequence when applying process reward?
            # - If we separate the retrieved facts and the retrieval analysis, the right retrieved facts and a **non-related** analysis might be preserved.
            # - The logic might be damaged.
            # Therefore, the step reward should be designed in fine-grained pattern.
            # --> Design a hierarchichal reward for the retrieved facts - analysis decision step.
            # *** How to combine this reward with the final total reward?
            # --> If the format and the answer are bad, if the retrieved facts are good, 
            global_score = curr_score_mean - global_score_mean
            global_score = 0.
            # TreeRL local score
            parent_leaf_num = node.parent_node.get_subtree_leaves_num()
            parent_score_mean = node.parent_node.subtree_leaf_score / parent_leaf_num
            local_score = curr_score_mean - parent_score_mean
            
            diff_score = global_score + local_score
            diff_score = max(diff_score - self.margin, 0.)

            # final score = diff_score + curr_score_mean
            final_score = diff_score + curr_score_mean
            node.final_score = final_score / math.sqrt(curr_leaf_num)

    def get_token_level_score_from_leaf(self):
        """
        Get the token-level score from the leaf
        """
        final_token_level_scores = torch.zeros_like(self.responses, dtype=torch.float32)
        
        valid_response_length = self.tensor_fn.create_attention_mask(self.responses).sum()
        final_token_level_scores[valid_response_length-1] = self.original_score

        return final_token_level_scores

    def extract_facts_info(self, responses_str):
        """Extract all <fact>...</fact> inside <retrieved_facts>...</retrieved_facts>"""
        af_block_pattern = r'<retrieved_facts>(.*?)</retrieved_facts>'
        fact_match = [f.strip() for f in re.findall(af_block_pattern, responses_str, re.DOTALL)]
        facts_info = '\n'.join(fact_match).strip()
        return fact_match

    def extract_information_list(self, responses_str):
        """Extract and concatenate information from <documents> tags, skipping the first."""
        info_pattern = r'<documents>(.*?)</documents>'
        matches = re.findall(info_pattern, responses_str, re.DOTALL)
        doc_info = '\n'.join(matches).strip()
        return matches

    def extract_search_queries(self, responses_str):
        """Extract all <search>...</search> queries in order"""
        query_pattern = r'<search>(.*?)</search>'
        queries = [q.strip() for q in re.findall(query_pattern, responses_str, re.DOTALL)]
        q_info = '\n'.join(queries).strip()
        return queries
    
    def normalize_text(self, text):
        """标准化文本：转小写、去标点、分词"""
        text = text.lower()
        # 去除标点
        text = ''.join(ch for ch in text if ch not in string.punctuation)
        # 分词（按空格，可替换为更高级分词）
        tokens = text.split()
        return tokens
    
    def is_valid_sequence(self, text):
        # # Find the position of "<|im_start|>assistant" with potential whitespace
        # assistant_pattern = r"<\|im_start\|>assistant\s*"
        # assistant_match = re.search(assistant_pattern, text)
        
        # if not assistant_match:
        #     return False, "Missing assistant marker"
        
        # # Extract the content after the assistant marker
        # start_pos = assistant_match.end()
        # content = text[start_pos:]
        content = text

        if '- [' not in text:
            return False, 'No Solutions.'
        
        # Check for balanced tags
        tags_to_check = ["search", "retrieved_facts", "answer", "analysis"]
        for tag in tags_to_check:
            opening_count = len(re.findall(f"<{tag}>", content))
            closing_count = len(re.findall(f"</{tag}>", content))
            if opening_count != closing_count:
                return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
        
        # Now check for proper sequence pattern and no extraneous content
        
        # 1. First split the content by any tags we recognize
        split_pattern = r"(</?(?:search|documents|retrieved_facts|analysis|answer)>)"
        parts = re.split(split_pattern, content)
        
        # 2. Keep track of the current position in the expected sequence
        state = "start"  # start -> think -> search -> retrieved_facts -> think -> ... -> answer -> end
        
        # 3. Check each part
        for i, part in enumerate(parts):
            # Skip empty parts
            if not part.strip():
                continue
                
            # Check if this is a tag
            if re.match(r"</?(?:search|documents|retrieved_facts|analysis|answer)>", part):
                if part == "<search>" and state in ["start"]:
                    state = "in_search"
                elif part == "</search>" and state == "in_search":
                    state = "after_search"
                elif part == "<documents>" and state == "after_search":
                    state = "in_doc"
                elif part == "</documents>" and state == "in_doc":
                    state = "out_doc"
                elif part == "<retrieved_facts>" and state == "out_doc":
                    state = "in_retrieved_facts"
                elif part == "</retrieved_facts>" and state == "in_retrieved_facts":
                    state = "retrieved_facts"
                elif part == "<analysis>" and state == "retrieved_facts":
                    state = "in_analysis"
                elif part == "</analysis>" and state == "in_analysis":
                    state = "out_analysis"
                elif part == "<answer>" and state == "out_analysis":
                    state = "in_answer"
                elif part == "<search>" and state == "out_analysis":
                    state = "in_search"
                elif part == "</answer>" and state == "in_answer":
                    state = "end"
                else:
                    return False, f"Unexpected tag {part} in state {state}"
            else:
                # This is content, check if it's valid in the current state
                # pass
                if state in ["start", "in_search", "after_search", "in_doc", "out_doc", "in_retrieved_facts", "retrieved_facts", "in_analysis", "in_answer"]:
                    # Content is allowed inside tags
                    pass
                elif state == "out_analysis" and ("<add_constraints>" in part or "<require>" in part):
                    pass
                elif i+1 < len(parts):
                    if "answer" in parts[i+1]:
                        pass
                    else:
                        return False, f"Need <require> or <add_constraints> tag."
                # elif state in ["start", "after_search", "retrieved_facts"]:
                #     # Only whitespace is allowed between tags
                #     if part.strip():
                #         return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
                else:
                    return False, f"Unexpected content in state {state}"
        
        # Check final state
        if state != "end":
            return False, f"Incomplete sequence, ended in state {state}"
        # elif state != "after_search":
        #     return False, f"Incomplete sequence, ended in state {state}"
            
        return True, "Valid sequence format"

    def vocab_overlap_reward(self, query: str, facts: str, docs: str, use_stopwords=False):
        """
        计算 reward = average(
            |words(facts) ∩ words(query)| / |words(query)|,
            |words(facts) ∩ words(docs)|   / |words(facts)|
        )
        
        Args:
            query (str): 用户问题
            facts (str): 模型提取的原子事实
            docs (str): 检索到的原始文档（可为多文档拼接）
            use_stopwords (bool): 是否过滤停用词（当前未实现，留作扩展）
        
        Returns:
            float: reward ∈ [0, 1]
        """
        if not query.strip() or not facts.strip():
            return 0.0

        query_tokens = set(self.normalize_text(query))
        facts_tokens = set(self.normalize_text(facts))
        docs_tokens = set(self.normalize_text(docs)) if docs.strip() else set()

        # 避免除零
        query_len = len(query_tokens) if query_tokens else 1
        facts_len = len(facts_tokens) if facts_tokens else 1

        # 1. Fact 覆盖 Query 的词汇比例
        overlap_qf = len(facts_tokens & query_tokens)
        coverage_query = overlap_qf / query_len

        # 2. Fact 被 Docs 支持的词汇比例
        overlap_fd = len(facts_tokens & docs_tokens)
        support_docs = overlap_fd / facts_len

        # 平均
        reward = (coverage_query + support_docs) / 2.0

        return min(max(reward, 0.0), 1.0)  # 确保在 [0,1]

    # def get_token_level_score_from_leaf_process(self, tokenizer):
    #     """
    #     Assign token-level rewards to each <fact> block inside <retrieved_facts>.
    #     Each fact i gets reward computed from query[i] and doc[i].
    #     Requires:
    #     - valid sequence format (is_valid_sequence)
    #     - len(queries) == len(docs) == len(facts)
    #     All non-fact tokens remain NaN.
    #     """
    #     device = self.responses.device
    #     seq_len = self.responses.size(0)
    #     # scores = torch.full((seq_len,), float('nan'), dtype=torch.float32, device=device)
    #     scores = torch.zeros_like(scores, dtype=torch.float32, device=device)

    #     response_str = tokenizer.decode(self.responses, skip_special_tokens=True)

    #     # ✅ Step 1: Format validation
    #     flag, _ = self.is_valid_sequence(response_str)
    #     if not flag:
    #         return scores  # all NaN

    #     # ✅ Step 2: Extract parallel lists
    #     queries = self.extract_search_queries(response_str)      # list of str
    #     documents = self.extract_information_list(response_str) # list of str
    #     facts = self.extract_facts_info(response_str)           # list of str

    #     if not (len(queries) == len(documents) == len(facts)) or len(facts) == 0:
    #         return scores  # length mismatch or empty

    #     # ✅ Step 3: For each fact, compute reward and locate tokens
    #     response_tokens = self.responses  # (seq_len,)

    #     for i in range(len(facts)):
    #         fact_str = facts[i].st)
    #         if not fact_str:rip(
    #             continue

    #         pattern = r'<retrieved_facts>(.*?)</retrieved_facts>'

    #         matches = list(re.finditer(pattern, response_str, re.DOTALL))

    #         match = matches[i]
    #         content_start, content_end = match.span(1)

    #         # ✅ Compute reward for this fact using its own query and doc
    #         query_i = queries[i].strip()
    #         doc_i = documents[i].strip() if documents[i].strip() else ''
    #         fact_i = fact_str

    #         # Skip if query is empty
    #         if not query_i:
    #             continue

    #         overlap_reward = self.vocab_overlap_reward(
    #             query=query_i,
    #             facts=fact_i,
    #             docs=doc_i,
    #             use_stopwords=False
    #         )
    #         scaled_reward = float(overlap_reward / 2.0)  # [0, 0.5]

    #         # ✅ Assign reward to matched token positions
    #         scores[content_start:content_end] = scaled_reward

    #     return scores

    def get_token_level_score_from_leaf_process(self, tokenizer):
        """
        Get the token-level score from the leaf
        """
        final_token_level_scores = torch.zeros_like(self.responses, dtype=torch.float32)
        response_str = tokenizer.decode(self.responses, skip_special_tokens=True)

        # Diff-based Reward
        if self.reward_mode == 'tree_diff':
            valid_response_length_list = []
            scores_list = []
            node = self.parent_node
            while node:
                if node.is_root:
                    break
                valid_response_length = self.tensor_fn.create_attention_mask(node.responses).sum()
                valid_response_length_list.append(valid_response_length)
                scores_list.append(node.final_score)
                node = node.parent_node
            scores_list.append(0.)
            valid_response_length_list.append(0)
            valid_response_length_list.reverse()
            scores_list.reverse()
            for i in range(1, len(valid_response_length_list)):
                score = scores_list[i]
                
                l = valid_response_length_list[i-1]
                r = valid_response_length_list[i]
                # r = valid_response_length_list[i] - 1

                if l < r:
                    final_token_level_scores[l:r] = score
        # Tree-GRPO uses base mode
        else:
            flag, _ = self.is_valid_sequence(response_str)
            if flag:
                retrieved_docs = self.extract_information_list(response_str) or []
                facts_list = self.extract_facts_info(response_str)
                search_queries = self.extract_search_queries(response_str)
                afr_score = 0.0

                if len(search_queries) == len(facts_list) and len(search_queries) == len(retrieved_docs) and len(search_queries) > 0:
                    for i in range(len(search_queries)):
                        s = self.vocab_overlap_reward(query=search_queries[i], facts=facts_list[i], docs=retrieved_docs[i])
                        afr_score += s
                    afr_score = afr_score / len(search_queries)
                valid_response_length = self.tensor_fn.create_attention_mask(self.responses).sum()
                final_token_level_scores[valid_response_length-1] = afr_score
            else:
                valid_response_length = self.tensor_fn.create_attention_mask(self.responses).sum()
                final_token_level_scores[valid_response_length-1] = 0

        return final_token_level_scores

    def _prune_subtree(self, candidate_uid_set: List[int]) -> bool:
        """
        Drop all the leaves not in cdandidate_uid_set
        """
        surviving_children = []
        dprint(f'start, node={self.node_uid}, child_node len={len(self._child_node)}')

        for child in self.child_node:
            dprint(f'node={self.node_uid}, iter for child={child.node_uid}')
            if child._prune_subtree(candidate_uid_set):
                surviving_children.append(child)
        
        # update child node
        self._child_node = surviving_children
                
        # Determine whether the current node should be retained. The conditions for retention are:
        # 1. It is in candidate_uid_set
        # 2. Or, it still has child nodes after pruning
        should_keep_this_node = self.node_uid in candidate_uid_set or (len(self._child_node) > 0)

        dprint(f'end, node={self.node_uid}, is keeped={should_keep_this_node}')

        return should_keep_this_node

    def check_all_nodes_child(self):
        for node in self.get_subtree_nodes():
            node_child_list = node.child_node
            node_child_node_uid_list = []
            for node_child_node in node_child_list:
                node_child_node_uid_list.append(node_child_node.node_uid)
            if node.node_uid in node_child_node_uid_list:
                dprint(f'error!! node.uid in node_child_node_uid_list!!! is_root={node.is_root}, is_leaf={node.is_leaf}, node_uid={node.node_uid}')

    def delete_tree_from_root(self):
        """
        Delete the tree from root
        """
        all_nodes_list = self.get_subtree_nodes()
        all_nodes_list.append(self)

        for node in all_nodes_list:
            node.prompts = None
            node.input_ids = None
            node.attention_mask = None
            node.position_ids = None
            node.responses = None
            node.responses_with_info_mask = None

            node._child_node.clear()
            node.parent_node = None

            node.log_prob_list.clear()

        