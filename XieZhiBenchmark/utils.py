"""

"""

import csv
import json
import random
import os
from tqdm import tqdm

random.seed(42)


def get_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def get_subdirectories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def sample_together(datas_list, num):
    """datas_list下每个list一样大, 按照一样的index来采样"""
    # 创建一个索引列表
    index_list = list(range(len(datas_list[0])))

    # 使用shuffle函数打乱索引列表
    random.shuffle(index_list)
    return [ [datas[index] for index in index_list[:num]] for datas in datas_list ], [ [datas[index] for index in index_list[num:] ] for datas in datas_list ]


def sample(datas, num):
    # 创建一个索引列表
    index_list = list(range(len(datas)))

    # 使用shuffle函数打乱索引列表
    random.shuffle(index_list)
    return [ datas[index] for index in index_list[:num] ], [ datas[index] for index in index_list[num:] ] 

def deduplicat(datas):
    """根据question去重"""
    inputs = []
    results = []
    for data in tqdm(datas):
        if data['question'] not in inputs:
            inputs.append(data['question'])
            results.append(data)
    return results


def del5(data):
     # 创建一个不包含正确答案的选项列表
    choices_without_correct = [i for i in range(len(data['choices'])) if i != data['answer']]
    
    # 从这个列表中随机选择一个选项进行删除
    del_index = random.choice(choices_without_correct)
    
    # 将正确答案的内容放到要删除的选项的位置
    data['choices'][del_index] = data['choices'][data['answer']]
    
    # 删除原来正确答案的位置的内容
    del data['choices'][data['answer']]
    
    if data['answer'] < del_index:
        # 更新正确答案的索引
        data['answer'] = del_index - 1
    else:
        data['answer'] = del_index

    return data


def read_csv_to_list(file_path):
    """
    读取csv文件并转换为列表
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        # 跳过标题行
        next(reader)
        # 将每一行转换为列表并存储
        data = [row for row in reader]
    return data

def ensure_file(path):
    # 确保文件路径中的目录存在
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_to_jsonl(datas, file_path):
    ensure_file(file_path)
    # datas = deduplicat(datas)
    with open(file_path, 'w') as f:
        for row in datas:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write('\n')

def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def read_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def insert_column(jsonls, column_name, column_data):
    for jsonl, data in zip(jsonls, column_data):
        jsonl[column_name] = data
    return jsonls


def save_txt(string, file_path):
    ensure_file(file_path)
    with open(file_path, 'w') as f:
        f.write(string)
