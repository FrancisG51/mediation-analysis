import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
import numpy as np


def bootstrap_function():

    concat_df = pd.read_csv('dataset/mediation_result.csv')
    concat_df['bootstrap下限'] = pd.NA
    concat_df['bootstrap上限'] = pd.NA
    data_all = pd.read_csv('dataset/data_all.csv')

    zibianliang = ['father depression', 'mother depression']
    xiebianliang = ['demo_prnt_ed_v2', 'demo_brthdat_v2', 'demo_comb_income_v2',
                    'demo_race_a_p___10', 'demo_race_a_p___11', 'male']

    def bootstrap(X, Y, M):

        meds = []
        retry_max = 200
        sample_n = 10000

        for _ in tqdm(range(sample_n)):
            retry = 0
            while retry < retry_max:
                try:
                    data = data_all.sample(data_all.shape[0], replace=True)

                    zibianliang_num4 = xiebianliang + zibianliang
                    zibianliang_num4 = data[zibianliang_num4]
                    zibianliang_num4 = sm.add_constant(zibianliang_num4)
                    zhongjiebianliang = data[M]
                    ols_mod = sm.OLS(zhongjiebianliang, zibianliang_num4).fit(maxiter=20, disp=False)

                    zibianliang_num5 = xiebianliang + zibianliang
                    zibianliang_num5.append(M)
                    zibianliang_num5 = data[zibianliang_num5]
                    zibianliang_num5 = sm.add_constant(zibianliang_num5)
                    yinbianliang = data[Y]
                    logit_mod = sm.Logit(yinbianliang, zibianliang_num5).fit(maxiter=20, disp=False)

                    meds.append(ols_mod.params[X] * logit_mod.params[M])
                    break
                except Exception as e:
                    retry = retry + 1

        # 跳出循环开始对所有的间接效应的值进行处理
        meds.sort()
        upper = int(sample_n * 0.975)
        lower = int(sample_n * 0.025)
        result_list = [meds[lower], meds[upper]]
        print('有效的bootstrap抽样的次数', len(meds))

        return result_list

    # 开始进行bootstrap检验
    for i in range(concat_df.shape[0]):
        print('第几个bootstrap检验', i + 1)
        X = concat_df.loc[i, '自变量']
        M = concat_df.loc[i, '中介变量']
        Y = concat_df.loc[i, '因变量']
        bootstrap_list = bootstrap(X, Y, M)
        concat_df.loc[i, 'bootstrap下限'] = bootstrap_list[0]
        concat_df.loc[i, 'bootstrap上限'] = bootstrap_list[1]

    concat_df['bootstrap下限'] = concat_df['bootstrap下限'].astype('float64')
    concat_df['bootstrap上限'] = concat_df['bootstrap上限'].astype('float64')
    concat_df = concat_df.round(4)
    concat_df.reset_index(drop=True, inplace=True)

    for i in range(concat_df.shape[0]):
        yihao_tonghao = concat_df.loc[i, 'bootstrap下限'] * concat_df.loc[i, 'bootstrap上限']
        if yihao_tonghao >= 0:
            concat_df.loc[i, 'bootstrap是否通过'] = '通过bootstrap检验'
        else:
            concat_df.loc[i, 'bootstrap是否通过'] = '不通过bootstrap检验'

    concat_df['是否为完全中介'] = np.where(concat_df['（一起回归）自变量到因变量，P值'] <= 0.05, '不完全中介', '完全中介')
    concat_df.reset_index(drop=True, inplace=True)
    concat_df.to_csv('dataset/bootstrap_result.csv', index=False)


def process_bootstrap_result():
    concat_df = pd.read_csv('dataset/bootstrap_result.csv')

    for i in range(concat_df.shape[0]):
        ab = concat_df.loc[i, '中介效应值（间接效应）']
        c = concat_df.loc[i, '（单一回归，自变量到因变量）系数（总效应）']
        c_pie = concat_df.loc[i, '（一起回归）自变量到因变量，系数（直接效应）']
        if ab * c_pie > 0:
            concat_df.loc[i, '中介效应占比'] = ab / c
            concat_df.loc[i, '是否是遮掩效应'] = '不是遮掩效应'
        else:
            concat_df.loc[i, '中介效应占比'] = abs(ab / c_pie)
            concat_df.loc[i, '是否是遮掩效应'] = '是遮掩效应'

    concat_df = concat_df.round(4)
    column_names = ['主效应', '中介变量', 'path-a', 'path-b', 'direct effect', 'indirect effect', 'indirect proportion',
                    'bootstrap pass']
    df_result = pd.DataFrame(columns=column_names)
    nan_hang = [np.nan] * len(column_names)
    df_result.loc[0] = nan_hang

    for i in range(concat_df.shape[0]):
        zibianliang = concat_df.loc[i, '自变量']
        yinbianliang = concat_df.loc[i, '因变量']
        df_result.loc[i, '主效应'] = zibianliang + ' to ' + yinbianliang

        df_result.loc[i, '中介变量'] = concat_df.loc[i, '中介变量']

        path_a_xishu = str(round(concat_df.loc[i, '（单一回归，自变量到中介变量）系数'], 4))
        path_a_p = str(round(concat_df.loc[i, '（单一回归，自变量到中介变量）P值'], 4))
        path_a_lower = str(round(concat_df.loc[i, '（单一回归，自变量到中介变量）置信下限'], 4))
        path_a_upper = str(round(concat_df.loc[i, '（单一回归，自变量到中介变量）置信上限'], 4))
        df_result.loc[i, 'path-a'] = ('β=' + path_a_xishu + ', p=' + path_a_p +
                                      ', CI=[' + path_a_lower + ', ' + path_a_upper + ']')

        path_b_xishu = str(round(concat_df.loc[i, '（一起回归）中介变量到因变量，系数'], 4))
        path_b_p = str(round(concat_df.loc[i, '（一起回归）中介变量到因变量，P值'], 4))
        path_b_lower = str(round(concat_df.loc[i, '（一起回归）中介变量到因变量，置信下限'], 4))
        path_b_upper = str(round(concat_df.loc[i, '（一起回归）中介变量到因变量，置信上限'], 4))
        df_result.loc[i, 'path-b'] = ('β=' + path_b_xishu + ', p=' + path_b_p +
                                      ', CI=[' + path_b_lower + ', ' + path_b_upper + ']')

        direct_effect_xishu = str(round(concat_df.loc[i, '（一起回归）自变量到因变量，系数（直接效应）'], 4))
        direct_effect_p = str(round(concat_df.loc[i, '（一起回归）自变量到因变量，P值'], 4))
        direct_effect_lower = str(round(concat_df.loc[i, '（一起回归）自变量到因变量，置信下限'], 4))
        direct_effect_upper = str(round(concat_df.loc[i, '（一起回归）自变量到因变量，置信上限'], 4))
        df_result.loc[i, 'direct effect'] = ('β=' + direct_effect_xishu + ', p=' + direct_effect_p +
                                             ', CI=[' + direct_effect_lower + ', ' + direct_effect_upper + ']')

        indirect_effect = str(round(concat_df.loc[i, '中介效应值（间接效应）'], 4))
        bootstrap_lower = str(round(concat_df.loc[i, 'bootstrap下限'], 4))
        bootstrap_upper = str(round(concat_df.loc[i, 'bootstrap上限'], 4))
        df_result.loc[i, 'indirect effect'] = indirect_effect + ', [' + bootstrap_lower + ', ' + bootstrap_upper + ']'

        df_result.loc[i, 'indirect proportion'] = str(round(concat_df.loc[i, '中介效应占比'], 4) * 100) + '%'
        df_result.loc[i, 'bootstrap pass'] = concat_df.loc[i, 'bootstrap是否通过']

        df_result.to_excel('dataset/final_result.xlsx', index=False)








