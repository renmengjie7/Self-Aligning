"""
读取
/home/XiezhiBenchmark/Tasks/Knowledge/Benchmarks/test/xiezhi_inter_eng/xiezhi.v1.1.json
/home/XiezhiBenchmark/Tasks/Knowledge/Benchmarks/test/xiezhi_spec_eng/xiezhi.v1.1.json
两个文件, 统计每个domain的数量
"""
import os
os.environ['PYTHONPATH'] = '/home/XiezhiBenchmark/mjren'
from utils import read_from_jsonl, write_to_jsonl


def cal(datas: list)-> dict:
    results = {}
    for data in datas:
        for label in data['labels']:
            if label in results.keys(): 
                results[label] += 1
            else:
                results[label] = 1
    return results


def cal_domain():
    files = [
        '/home/XiezhiBenchmark/Tasks/Knowledge/Benchmarks/test/xiezhi_inter_eng/xiezhi.v1.1.json',
        '/home/XiezhiBenchmark/Tasks/Knowledge/Benchmarks/test/xiezhi_spec_eng/xiezhi.v1.1.json',
    ]
    # {'History': 8958, 'Jurisprudence': 6921, 'Political Science': 5266, 'Engineering': 5089, 'Science': 3872, 'Economics': 3249, 'Literature': 2496, 'Medicine': 2212, 'History (Level 1 subject)': 2165, 'Philosophy': 2088, ...}
    files=[
        '/home/XiezhiBenchmark/Tasks/Knowledge/OtherXiezhiData_Noisy_ReviewOnly/xiezhi_all_noisy/xiezhi.v1.json'
    ]
    # {'历史学': 162353, '法学': 52188, '经济学': 44449, '哲学': 40365, '政治学': 36638, '工学': 35917, '文学': 26971, '理学': 18563, '历史学（1级学科）': 11973, '医学': 11238, '农学': 11148, '军事学': 7510, '教育学': 6984, '中国史': 6732, ...}
    results = {}
    for file in files:
        temp = cal(read_from_jsonl(file))
        for key in temp.keys():
            if key in results.keys():
                results[key] += temp[key]
            else:
                results[key] = temp[key]
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    print(results)
    

def split_domain():
    result_root = '/home/XiezhiBenchmark/mjren/result/source/benchmark_en'
    files = [
        '/home/XiezhiBenchmark/Tasks/Knowledge/Benchmarks/test/xiezhi_inter_eng/xiezhi.v1.1.json',
        '/home/XiezhiBenchmark/Tasks/Knowledge/Benchmarks/test/xiezhi_spec_eng/xiezhi.v1.1.json',
    ]
    results = {'History': [], 'Science': [], 'Engineering': [], 'Jurisprudence': []}
    for file in files:
        datas = read_from_jsonl(file)
        for data in datas:
            for label in data['labels']:
                if label in results.keys():
                    results[label].append(data)
    for domain in results.keys():
        write_to_jsonl(results[domain], os.path.join(result_root, domain+'.jsonl'))
    
def main():
    # cal_domain()
    split_domain()


if __name__ == "__main__":
    main()
    
