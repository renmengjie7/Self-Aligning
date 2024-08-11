"""
将xiezhi转成可以由domain_eval读入的格式

source
{"question": "Since the 1990s, _____ has become a leading industry in the U.S. economy", "labels": ["Applied Economics", "Industrial Economics", "Economics", "Engineering"], "answer": "High-tech industry", "options": "industry\nagriculture\nTertiary industry\nHigh-tech industry"}

target
{"id": 39426, "question": "除哪项外均是枕先露分娩机转的动作", "choices": ["衔接", "下降", "拨露", "仰伸", "外旋转"], "answer": 2, "explanation": "", "raw": {"exam_type": "医师考试", "exam_class": "执业医师", "exam_subject": "公共卫生执业医师", "question": "除哪项外均是枕先露分娩机转的动作", "answer": "C", "question_type": "单项选择题", "option": {"A": "衔接", "B": "下降", "C": "拨露", "D": "仰伸", "E": "外旋转"}}}
"""
from utils import read_from_jsonl, write_to_jsonl
import os
import random
random.seed(42)


def split():
    """拆分train和test(留个250条吧😂)"""
    source_root = '/home/XiezhiBenchmark/mjren/result/processed/benchmark_en'
    target_root = '/home/XiezhiBenchmark/mjren/result/split/benchmark_en'
    domains = ['Engineering', 'History', 'Science', 'Jurisprudence']
    for domain in domains:
        file = f'{domain}.jsonl'
        datas = read_from_jsonl(os.path.join(source_root, file))
        # random
        random.shuffle(datas)
        test = datas[:250]
        dev = datas[250:260]
        train = datas[260:]
        # 保存
        os.makedirs(f'{target_root}/{file.replace(".jsonl", "").lower()}', exist_ok=True)
        write_to_jsonl(train, f'{target_root}/{file.replace(".jsonl", "").lower()}/train.jsonl')
        write_to_jsonl(dev, f'{target_root}/{file.replace(".jsonl", "").lower()}/dev.jsonl')
        write_to_jsonl(test, f'{target_root}/{file.replace(".jsonl", "").lower()}/test.jsonl')


def main():
    source_root = '/home/XiezhiBenchmark/mjren/result/source/benchmark_en'
    target_root = '/home/XiezhiBenchmark/mjren/result/processed/benchmark_en'
    for file in [item for item in os.listdir(source_root) if item.endswith('jsonl')]:
        datas = read_from_jsonl(os.path.join(source_root, file))
        results = []
        # 转换
        for idx, data in enumerate(datas):
            result = {}
            result['id'] = idx
            result['question'] = data['question']
            result['choices'] = data['options'].split('\n')
            if len(result['choices']) !=4:
                print(result['choices'])
                continue
            result['answer'] =result['choices'].index(data['answer'])
            result['explanation'] = ''
            result['raw'] = data
            results.append(result)
        # 保存
        write_to_jsonl(results, os.path.join(target_root, file))


if __name__ == "__main__":
    # main()
    split()
