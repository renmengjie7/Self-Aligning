import numpy as np
from scipy.stats import entropy
import sys
sys.path.append('/Users/bubble/Desktop/mine/self-align/提交/code/eval')
import os
import numpy as np
from utils import read_from_jsonl
from scipy.stats import pearsonr
from categories import mmlu_subcategories
from categories import cmmlu_subcategories, cmmlu_subcategories, mmlu_categories, mmlu_subcategories
from numpy import dot
from numpy.linalg import norm

mmlu_domain={
    'math': [ key for key in mmlu_subcategories.keys() if mmlu_subcategories[key] == ['math'] ],
    'medical': [ key for key in mmlu_subcategories.keys() if mmlu_subcategories[key] == ['health'] ],
    'medmcqa': [ key for key in mmlu_subcategories.keys() if mmlu_subcategories[key] == ['health'] ],

    'law': [ key for key in mmlu_subcategories.keys() if mmlu_subcategories[key] == ['law'] ],
    'jurisprudence': [ key for key in mmlu_subcategories.keys() if mmlu_subcategories[key][0] in ["politics", "culture", "economics", "geography", "psychology", 'law']],

    'engineering': [ key for key in mmlu_subcategories.keys() if mmlu_subcategories[key] == ['engineering'] ],
    'science': [ key for key in mmlu_subcategories.keys() if mmlu_subcategories[key] == ['computer science'] ],

    'history': [ key for key in mmlu_subcategories.keys() if mmlu_subcategories[key] == ['history'] ]
}

cmmlu_domain={
    'math': [ key for key in cmmlu_subcategories.keys() if cmmlu_subcategories[key] == ['math'] ],
    'medical': [ 'anatomy', 'clinical_knowledge', 'college_medicine', 
                'nutrition', 'professional_medicine', 'virology' ],
    'law': [ key for key in cmmlu_subcategories.keys() if cmmlu_subcategories[key] == ['law'] ],
}

ceval_domain={
    'math': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics','probability_and_statistics'],
    'medical': ['clinical_medicine', 'physician', 'veterinary_medicine', 'basic_medicine', 'veterinary_medicine'],
    'law': ['law'],
}


def mmlu_probs(path, domains=None):
    """读取result.txt的最后一行"""
    def find_others(domains):
        others = []
        for item in mmlu_subcategories.keys():
            add=True
            for domain in domains:
                if item in mmlu_domain[domain]:
                    add=False
                    break
            if add:
                others.append(item)
        return others
      
    def read_jsonl_probs(path):
        datas = read_from_jsonl(path)
        return [item['probs'] for item in datas]
    
    def get_probs(root, files):
        probs = []
        for file in files:
            probs.extend(read_jsonl_probs(os.path.join(root, f'{file}.jsonl')))
        return probs

    return get_probs(f'{path}/split', find_others(domains)), [ get_probs(f'{path}/split', mmlu_domain[domain]) for domain in domains]


def replace_with_order(input_list):
    sorted_values = sorted(set(input_list), reverse=True)
    value_to_order = {value: order for order, value in enumerate(sorted_values, start=1)}
    output_list = [value_to_order[value] for value in input_list]
    return output_list


def cosine_similarity(list1, list2):
    """
    计算两个列表的余弦相似性。

    参数:
    list1 -- 第一个向量，列表或者NumPy数组形式。
    list2 -- 第二个向量，列表或者NumPy数组形式。

    返回:
    similarity -- 两个向量的余弦相似性。
    """
    # 将列表转换为NumPy数组
    vector_a = np.array(list1)
    vector_b = np.array(list2)
    
    # 计算两个向量的点积
    dot_product = dot(vector_a, vector_b)
    
    # 计算两个向量的范数（长度）
    norm_a = norm(vector_a)
    norm_b = norm(vector_b)
    
    # 计算余弦相似性
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity


def kl_divergence(p, q):
    """
    计算两个概率分布p和q之间的KL散度。
    
    参数:
    p -- 第一个概率分布（实际分布），必须是一个有效的概率分布列表或者NumPy数组。
    q -- 第二个概率分布（理论分布），必须是一个有效的概率分布列表或者NumPy数组。
    
    返回:
    kl_div -- p和q之间的KL散度。
    """
    # 验证p和q是否是有效的概率分布
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # 确保两个分布的长度相同
    if p.shape != q.shape:
        raise ValueError("两个分布的长度必须相同。")
    
    # 确保分布是非负的
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("分布中不能有负值。")
    
    # 确保分布已经归一化
    if not np.isclose(np.sum(p), 1) or not np.isclose(np.sum(q), 1):
        raise ValueError("分布必须归一化，即总和为1。")
    
    # 计算KL散度
    kl_div = entropy(p, q)
    
    return kl_div


def cal(types, model, domain, split, test, probs1, probs2):
    if 'kl' in types:
        kl = sum([ kl_divergence(item1, item2) for item1, item2 in zip(probs1, probs2) ])/len(probs1)
        print(f'{model},{domain},{split},{test},kl,{kl}')
    if 'cosine' in types:
        sim = sum([ cosine_similarity(item1, item2) for item1, item2 in zip(probs1, probs2) ])/len(probs1)
        print(f'{model},{domain},{split},{test},cosine,{sim}')
    if 'pearson' in types:
        probs1 = [replace_with_order(item) for item in probs1]
        probs2 = [replace_with_order(item) for item in probs2]
        correlation_coefficient = sum([ calculate_pearson_correlation(item1, item2)[0] for item1, item2 in zip(probs1, probs2) ])/len(probs1)
        print(f'{model},{domain},{split},{test},pearson,{correlation_coefficient}')


def calculate_pearson_correlation(list1, list2):
    correlation_coefficient, p_value = pearsonr(list1, list2)
    if np.isnan(correlation_coefficient):
        correlation_coefficient = 0
    return correlation_coefficient, p_value


def cal_homo(model, domain, split, types):
    source_domain=f'domain-base/{model}/{domain}-logits-shot5-test-vicuna0.jsonl'
    result_domain = f'domain-sft/{model}-{domain}-{split}-bs256-ep3-ep3/{domain}-logits-shot0-test-vicuna1.jsonl'  

    datas1 = read_from_jsonl(source_domain)
    datas2 = read_from_jsonl(result_domain)
    
    probs1 = [item['probs'] for item in datas1]
    probs2 = [item['probs'] for item in datas2]

    cal(types, model, domain, split, 'homo', probs1, probs2)
    

def cal_ood(model, domain, split, types):
    
    source_mmlu = f'mmlu-base/{model}/logits-shot5-test-vicuna0'
    result_mmlu = f'mmlu-sft/{model}-{domain}-{split}-bs256-ep3-ep3/logits-shot0-test-vicuna1'
    
    ood1, id1 = mmlu_probs(source_mmlu, [domain])
    ood2, id2 = mmlu_probs(result_mmlu, [domain])
    
    probs1 = ood1
    probs2 = ood2
    
    cal(types, model, domain, split, 'ood', probs1, probs2)


def cal_id(model, domain, split, types):
    source_mmlu = f'mmlu-base/{model}/logits-shot5-test-vicuna0'
    result_mmlu = f'mmlu-sft/{model}-{domain}-{split}-bs256-ep3-ep3/logits-shot0-test-vicuna1'

    ood1, id1 = mmlu_probs(source_mmlu, [domain])
    ood2, id2 = mmlu_probs(result_mmlu, [domain])
    
    probs1 = id1[0]
    probs2 = id2[0]

    cal(types, model, domain, split, 'id', probs1, probs2)


def main():
    splits=[
        'inconsistent-golden-exp', 
        'consistent_wrong0.05-inconsistent_golden-exp', 
        'consistent_wrong0.1-inconsistent_golden-exp', 
        'consistent_wrong0.2-inconsistent_golden-exp', 
        'consistent_wrong0.4-inconsistent_golden-exp',
        'consistent_wrong0.6-inconsistent_golden-exp',
        'consistent_wrong0.8-inconsistent_golden-exp',
        'inconsistent-predict-model-exp',
    ]
    models=[
        'llama-2-7b', 
        'llama-2-13b', 
        'mistral-7b', 
        # 'llama-2-70b'
    ]
    domains = ['jurisprudence', 'history', 'engineering', 'medmcqa']
    print('model, domain, split, test, type, value')
    types = ['kl', 'pearson']
    for model in models:
        for domain in domains:
            for split in splits:
                if os.path.exists(f'domain-sft/{model}-{domain}-{split}-bs256-ep3/{domain}-logits-shot0-test-vicuna1.jsonl'):
                    cal_homo(model, domain, split, types)
                    cal_id(model, domain, split, types)
                    cal_ood(model, domain, split, types)


if __name__ == '__main__':
    main()
