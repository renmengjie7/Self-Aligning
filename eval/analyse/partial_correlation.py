"""
计算偏相关性

偏相关性是指在控制一个或多个变量的影响下，研究两个变量之间相关关系的强度和方向的统计指标。

X: Pearson correlation between fine-tuned model performance and base model performance on the test set
Y: fine-tuned model performance on the same test set
Z: base model performance on the same test
"""
import statsmodels.api as sm
from scipy.stats import spearmanr
import pandas as pd
from utils import csv_to_dict_list


def calculate_spearman_partial_correlation(data, var_x, var_y, control_vars):
    """
    计算Spearman偏相关系数。

    :param data: 包含分析变量的pandas DataFrame。
    :param var_x: 字符串，分析的第一个变量的名称。
    :param var_y: 字符串，分析的第二个变量的名称。
    :param control_vars: 字符串列表，控制变量的名称。
    :return: Spearman偏相关系数。
    """
    # 计算控制变量的秩
    for var in control_vars:
        data[var + '_rank'] = data[var].rank()

    # 将var_x和var_y也转换为秩
    data[var_x + '_rank'] = data[var_x].rank()
    data[var_y + '_rank'] = data[var_y].rank()

    # 对Y和控制变量的秩进行线性回归，得到残差
    Y_rank = data[var_y + '_rank']
    X_controls_rank = data[[var + '_rank' for var in control_vars]]
    X_controls_rank = sm.add_constant(X_controls_rank)  # 添加常数项
    model_Y_rank = sm.OLS(Y_rank, X_controls_rank).fit()
    residual_Y_rank = model_Y_rank.resid

    # 对X和控制变量的秩进行线性回归，得到残差
    X_rank = data[var_x + '_rank']
    model_X_rank = sm.OLS(X_rank, X_controls_rank).fit()
    residual_X_rank = model_X_rank.resid

    # 计算残差之间的Spearman相关系数
    spearman_corr, p_value = spearmanr(residual_Y_rank, residual_X_rank)

    return spearman_corr, p_value


def main():
    # 读取merge_base文件
    datas = csv_to_dict_list('merge.csv')
    print('model,test,corr,p')
    for model in ['mistral-7b', 'llama-2-7b', 'llama-2-13b']:
        merge_base = [item for item in datas if ('consistent_wrong' in item['split'] or item['split']=='inconsistent-predict-model-exp' or item['split'] == 'inconsistent-golden-exp') and item['model']==model]
  
        ratios = []
        for item in merge_base:
            if item['split'] == 'inconsistent-golden-exp':
                ratios.append(0)
            elif item['split'] == 'inconsistent-predict-model-exp':
                ratios.append(1)
            else:
                for ratio in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]:
                    if item['split'] == f'consistent_wrong{ratio}-inconsistent_golden-exp':
                        ratios.append(ratio)
                        break
        result_str = f'{model}  &   '
        for type in ['homo', 'mmlu-id', 'mmlu-ood']:               
            data = pd.DataFrame({
                'X': [ float(item[f'{type}-pearson']) * 100 for item in merge_base ],
                'Y': [ float(item[f"{type.replace('id1','id')}"]) for item in merge_base ],
                'Z1': [ float(item[f'base-{type}']) for item in merge_base ], 
            })
            partial_corr1, p = calculate_spearman_partial_correlation(data, 'X', 'Y', ['Z1'])
            result_str += f'{partial_corr1:.2f}    &     {p:.2f}    &      ' 
        result_str += '\\\\'
        print(result_str)


if __name__ == '__main__':
    main()

