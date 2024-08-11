import csv
import json
import random
import os
from tqdm import tqdm

random.seed(42)


def csv_to_dict_list(file_path):
    # 创建一个空列表来存储字典
    dict_list = []
    
    # 打开文件并读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        # 使用csv.DictReader读取CSV文件，它会使用第一行作为字典的键
        csv_reader = csv.DictReader(file)
        # 遍历CSV中的每一行，并将其添加到列表中
        for row in csv_reader:
            dict_list.append(row)
    
    return dict_list


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
    ensure_file(path)
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
