# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence_easy(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["search", "refine", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:search|documents|refine|analysis|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> refine -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:search|documents|refine|analysis|answer)>", part):
            # This is a tag, check if it's valid in the current state
            # if part == "<think>" and state in ["start", "refine"]:
            #     state = "in_think"
            # elif part == "</think>" and state == "in_think":
            #     state = "after_think"
            if part == "<search>" and state in ["start"]:
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<documents>" and state == "after_search":
                state = "in_doc"
            elif part == "</documents>" and state == "in_doc":
                state = "out_doc"
            elif part == "<refine>" and state == "out_doc":
                state = "in_refine"
            elif part == "</refine>" and state == "in_refine":
                state = "refine"
            elif "<analysis>" in part and state == "refine":
                state = "in_analysis"
            elif "</analysis>" in part and state == "in_analysis":
                state = "out_analysis"
            elif part == "<answer>" and state in ["out_analysis"]:
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            pass
            # This is content, check if it's valid in the current state
            if state in ["start", "in_search", "in_doc", "in_refine", "in_analysis", "in_answer"]:
                # Content is allowed inside tags
                pass
            # elif state in ["start", "after_think", "after_search", "refine"]:
            #     # Only whitespace is allowed between tags
            #     if part.strip():
            #         return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    # if state != "end":
    #     return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"

def is_valid_sequence(response):
    """
    validate the template format
    return: (is valid)
    """
    token_list = ['search', 'refine', 'answer']

    if not response:
        return False, "Inalid sequence format"
    
    for special_tags in token_list:
        start_token = f"<{special_tags}>"
        end_token = f"</{special_tags}>"
        start_count = response.count(start_token)
        end_count = response.count(end_token)
        if start_count != end_count:
            return False, "Inalid sequence format"
        if start_count == 0:
            return False, "Inalid sequence format"
        if 'refine' not in response:
            return False, "Inalid sequence format"
    # ans = extract_solution(response)
    # if 'Not Applicable' not in ans and response.endswith('</search>'):
    #     return 0

    return True, "Valid sequence format"


def validate_format_easy(response):
    """
    validate the template format
    return: (is valid)
    """
    token_list = ['search', 'refine', 'answer', 'analysis']

    if not response:
        return False
    
    for t in token_list:
        if t not in response:
            return False

    return True

def validate_format_med(response):
    """
    validate the template format
    return: (is valid)
    """
    token_list = ['search', 'refine', 'answer', 'analysis']

    if not response:
        return False
    
    for special_tags in token_list:
        start_token = f"<{special_tags}>"
        end_token = f"</{special_tags}>"
        start_count = response.count(start_token)
        end_count = response.count(end_token)
        # if start_count != end_count:
        #     return False
        if start_count == 0:
            return False
        if 'refine' not in response:
            return False
        if 'analysis' not in response:
            return False
    # ans = extract_solution(response)
    # if 'Not Applicable' not in ans and response.endswith('</search>'):
    #     return 0

    return True

def validate_format_hard(response):
    """
    validate the template format
    return: (is valid)
    """
    token_list = ['search', 'refine', 'answer', 'analysis']

    if not response:
        return False
    
    for special_tags in token_list:
        start_token = f"<{special_tags}>"
        end_token = f"</{special_tags}>"
        start_count = response.count(start_token)
        end_count = response.count(end_token)
        if start_count != end_count:
            return False
        if start_count == 0:
            return False
        if 'analysis' not in response:
            return False
        if '- [' not in response:
            return False
        # else:
        #     if '<add_constraints>' not in response or '<require>' not in response:
        #         return False
    # ans = extract_solution(response)
    # if 'Not Applicable' not in ans and response.endswith('</search>'):
    #     return 0

    return True

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def extract_solution(solution_str: str) -> str | None:
    """
    Extract the content inside the last <answer>...</answer> tag.
    Returns None if no such tag is found.
    """
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, solution_str, flags=re.DOTALL)
    
    if not matches:
        return None
    
    # re.findall returns list of strings (the captured group)
    # Take the last one and strip whitespace
    return matches[-1].strip()


def extract_refine_blocks(text: str) -> list[str]:
    pattern = r"<refine>(.*?)</refine>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_refine_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False

def extract_search_queries(responses_str):
    """Extract all <search>...</search> queries in order"""
    query_pattern = r'<search>(.*?)</search>'
    queries = [q.strip() for q in re.findall(query_pattern, responses_str, re.DOTALL)]
    return queries

def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1., data_source=''):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    search_queries = extract_search_queries(solution_str)
    is_valid_format, is_valid_reason = is_valid_sequence(solution_str)
    # is_valid_format = validate_format(solution_str)
    retrieval_correct = False
    if is_valid_format:
        is_valid_reason = "Valid sequence"
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    # else:
    #     is_valid_reason = "Lack the tags."
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    # do_print = True
    
    if do_print:
        print(f"--------------------------------")
        print(f"Data source: {data_source}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"is_valid_format: {is_valid_format}, reason: {is_valid_reason}")
        if answer is not None:
            print(f"em_check: {em_check(answer, ground_truth['target'])}")
        else:
            print(f"em_check: answer is None")
    
    # if validate_format_easy(solution_str):
    #     level1 = True
    # else:
    #     return 0
    
    # if level1:
    #     if validate_format_med(solution_str):
    #         level2 = True
    #     else:
    #         return 0.1
    
    # if level2:
    #     if validate_format_hard(solution_str):
    #         level3 = True
    #     else:
    #         return 0.2

    level3 = True
    
    if level3:
        bonus = 0
        if answer is None:
            if retrieval_correct:
                return 0
            else:
                return 0
        else:
            if em_check(answer, ground_truth['target']):
                if is_valid_format:
                    return score
                else:
                    if retrieval_correct:
                        return 0.8
                    else:
                        return 0
            elif is_valid_format:
                if retrieval_correct:
                    return 0.3
                else:
                    if len(search_queries) == 1:
                        final_reward = 0.1 # 0.1
                        return final_reward
                    else:
                        return 0.2
            else:
                return 0

    # if level3:
    #     bonus = 0.3
    #     if answer is None:
    #         if is_valid_format:
    #             if retrieval_correct:
    #                 # return structure_format_score + retrieval_score + bonus # 0.3+0.3=0.6
    #                 return 0.45
    #             else:
    #                 # return structure_format_score # 0.2+0.3=0.5
    #                 return 0.4
    #         else:
    #             return bonus # 0.3
    #     else:
    #         if em_check(answer, ground_truth['target']):
    #             if is_valid_format:
    #                 return score + bonus # 1.0+0.3=1.3
    #             else:
    #                 return score - structure_format_score + bonus # 0.8+0.3=1.1
    #         elif is_valid_format:
    #             if retrieval_correct:
    #                 return structure_format_score + retrieval_score + bonus # 0.3+0.3=0.6
    #             else:
    #                 if len(search_queries) == 1:
    #                     final_reward = 0.1 + bonus # 0.4
    #                     return final_reward
    #                 else:
    #                     return structure_format_score + bonus # 0.2+0.3=0.5
    #         else:
    #             return final_format_score + bonus # 0.1+0.3=0.4

def extract_atomic_facts(responses_str):
    """Extract all <fact>...</fact> inside <refine>...</refine>"""
    af_block_pattern = r'<refine>(.*?)</refine>'
    fact_match = [f.strip() for f in re.findall(af_block_pattern, responses_str, re.DOTALL)]
    facts = '\n'.join(fact_match).strip()
    return facts

def extract_facts_list(responses_str):
    """Extract all <fact>...</fact> inside <refine>...</refine>"""
    af_block_pattern = r'<refine>(.*?)</refine>'
    fact_match = [f.strip() for f in re.findall(af_block_pattern, responses_str, re.DOTALL)]
    return fact_match

def extract_retrieval_analysis(responses_str):
    """Extract content inside <analysis>...</analysis>"""
    ra_pattern = r'<analysis>(.*?)</analysis>'
    ra_match = [r.strip() for r in re.findall(ra_pattern, responses_str, re.DOTALL)]
    return ra_match

def extract_search_queries(responses_str):
    """Extract all <search>...</search> queries in order"""
    query_pattern = r'<search>(.*?)</search>'
    queries = [q.strip() for q in re.findall(query_pattern, responses_str, re.DOTALL)]
    return queries

def extract_facts_info(responses_str):
    """Extract all <fact>...</fact> inside <refine>...</refine>"""
    af_block_pattern = r'<refine>(.*?)</refine>'
    fact_match = [f.strip() for f in re.findall(af_block_pattern, responses_str, re.DOTALL)]
    facts_info = '  '.join(fact_match).strip()
    return facts_info

def extract_doc_info(responses_str):
    """Extract all <fact>...</fact> inside <refine>...</refine>"""
    af_block_pattern = r'<documents>(.*?)</documents>'
    fact_match = [f.strip() for f in re.findall(af_block_pattern, responses_str, re.DOTALL)]
    facts_info = '  '.join(fact_match).strip()
    return facts_info

def extract_information_list(responses_str):
    """Extract and concatenate information from <documents> tags, skipping the first."""
    info_pattern = r'<documents>(.*?)</documents>'
    matches = re.findall(info_pattern, responses_str, re.DOTALL)
    
    # if len(matches) <= 1:
    #     return None
    # matches = matches[1:]
    return matches

def vocab_overlap_reward(query: str, facts: str, docs: str, use_stopwords=False):
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

    query_tokens = set(normalize_text(query))
    facts_tokens = set(normalize_text(facts))
    docs_tokens = set(normalize_text(docs)) if docs.strip() else set()

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

def cover_em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def compute_score_format(responses_str, ground_truth):
    format_validity, _ = is_valid_sequence(responses_str)
    if format_validity:
        score = 1
    else:
        score = 0
    return score

def compute_score_cem(responses_str, ground_truth):
    answer = extract_solution(responses_str)
    if answer is None:
        return 0
    else:
        return cover_em_check(answer, ground_truth['target'])


def compute_information_score_subem(responses_str, ground_truth):
    information = extract_doc_info(responses_str)
    
    if information is None:
        return 0.0
    elif 'no' in ground_truth['target'] or 'yes' in ground_truth['target']:
        return 0.5
    else:
        return cover_em_check(information, ground_truth['target'])

def compute_information_reverse_rank(responses_str, ground_truth):
    doc_list = extract_information_list(responses_str)
    info_score = 0.0
    
    if doc_list is None:
        return 0.0
    elif 'no' in ground_truth['target'] or 'yes' in ground_truth['target']:
        return 0.5
    else:
        for idx, doc in enumerate(doc_list):
            if cover_em_check(doc, ground_truth['target']):
                info_score += 1 / float(idx + 1)
    return info_score

def get_shot_num(responses_str, ground_truth):
    or_score = compute_score_f1(responses_str, ground_truth)
    cem_score = compute_information_score_subem(responses_str, ground_truth)
    if or_score > 0.1 and cem_score > 0:
        return 1
    else:
        return 0
    
def get_success_constraint(responses_str, ground_truth):
    or_score = compute_score_f1(responses_str, ground_truth)
    if "<add_constraints>" in responses_str and or_score > 0.1:
        return 1
    else:
        return 0
    
def get_success_require(responses_str, ground_truth):
    or_score = compute_score_f1(responses_str, ground_truth)
    if "<require>" in responses_str and or_score > 0.1:
        return 1
    else:
        return 0
    
def get_constraint(responses_str, ground_truth):
    if "<add_constraints>" in responses_str:
        return 1
    else:
        return 0

def get_require(responses_str, ground_truth):
    if "<require>" in responses_str:
        return 1
    else:
        return 0
    
def normalize_text(text):
    """标准化文本：转小写、去标点、分词"""
    text = text.lower()
    # 去除标点
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    # 分词（按空格，可替换为更高级分词）
    tokens = text.split()
    return tokens

def compute_f1_scores(prediction: str, ground_truths: list):
    final_metric = {"f1": 0, "precision": 0, "recall": 0}
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        if (
            normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        for k in ["f1", "precision", "recall"]:
            final_metric[k] = max(eval(k), final_metric[k])
    return final_metric['f1']

def compute_score_f1(responses_str, ground_truth):
    answer = extract_solution(responses_str)
    if answer is None:
        return 0
    else:
        return compute_f1_scores(answer, ground_truth['target'])
    
def compute_em(responses_str, ground_truth):
    answer = extract_solution(responses_str)
    if answer is None:
        return 0
    else:
        return em_check(answer, ground_truth['target'])

def compute_score_em2(solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1., data_source=''):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    search_queries = extract_search_queries(solution_str)
    is_valid_format, is_valid_reason = is_valid_sequence(solution_str)
    # is_valid_format = validate_format(solution_str)
    retrieval_correct = False
    if is_valid_format:
        is_valid_reason = "Valid sequence"
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    # else:
    #     is_valid_reason = "Lack the tags."
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1,64) == 1

    # do_print = True
    
    if do_print:
        print(f"--------------------------------")
        print(f"Data source: {data_source}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"is_valid_format: {is_valid_format}, reason: {is_valid_reason}")
        if answer is not None:
            print(f"em_check: {em_check(answer, ground_truth['target'])}")
        else:
            print(f"em_check: answer is None")
    
    level3 = True
    
    if level3:
        bonus = 0
        if answer is None:
            if retrieval_correct:
                return 0.1
            else:
                return 0
        else:
            if em_check(answer, ground_truth['target']):
                if is_valid_format:
                    return score
            elif retrieval_correct:
                return 0.1
            else:
                return 0
