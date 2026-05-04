import pandas as pd
import copy
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


def statistical_function_logistic():

    data_all = pd.read_csv('dataset/data_all.csv')

    print('年龄列的均值：', round(data_all['demo_brthdat_v2'].mean(), 2))
    print('年龄列的标准差：', round(data_all['demo_brthdat_v2'].std(), 2))

    yinbianliang = ['adolescent MDD', 'adolescent UDD', 'adolescent MDS']
    zibianliang = ['mother depression', 'father depression']
    xiebianliang = ['demo_prnt_ed_v2', 'demo_brthdat_v2', 'demo_comb_income_v2',
                    'demo_race_a_p___10', 'demo_race_a_p___11', 'male']

    def main_effect_logistic(y):
        zibianliang_linshi = copy.deepcopy(zibianliang + xiebianliang)
        zibianliang_linshi = data_all[zibianliang_linshi]
        zibianliang_linshi = sm.add_constant(zibianliang_linshi)
        yinbianliang_linshi = copy.deepcopy(y)
        yinbianliang_linshi = data_all[yinbianliang_linshi]

        logit_mod = sm.Logit(yinbianliang_linshi, zibianliang_linshi).fit(maxiter=20, disp=False)
        ci_df = copy.deepcopy(logit_mod.conf_int())
        ci_df.rename(columns={ci_df.columns[0]: 'lower'}, inplace=True)
        ci_df.rename(columns={ci_df.columns[1]: 'upper'}, inplace=True)
        ci_df['means'] = (ci_df['upper'] + ci_df['lower']) / 2
        ci_df['自变量'] = ci_df.index
        ci_df['因变量'] = y
        ci_df = ci_df.loc[ci_df.index.isin(zibianliang), :]
        for i, variable in enumerate(list(ci_df.index)):
            ci_df.loc[variable, 'p值'] = logit_mod.pvalues[variable]
            ci_df.loc[variable, '系数'] = logit_mod.params[variable]
        ci_df = ci_df.round(4)
        return ci_df

    column_names = ['lower', 'upper', 'means', '自变量', '因变量', 'p值', '系数']
    concat_df = pd.DataFrame(columns=column_names)
    nan_hang = [np.nan] * len(column_names)
    concat_df.loc[0] = nan_hang

    for y in yinbianliang:
        concat_df = pd.concat([concat_df, main_effect_logistic(y)], axis=0)
    logit_ci_df = concat_df.dropna(thresh=int(1 * concat_df.shape[1]), axis=0)
    logit_ci_df = logit_ci_df.round(4)
    logit_ci_df.reset_index(drop=True, inplace=True)

    logit_ci_df['lower'] = np.exp(logit_ci_df['lower']).round(4)
    logit_ci_df['upper'] = np.exp(logit_ci_df['upper']).round(4)
    logit_ci_df['means'] = np.exp(logit_ci_df['means']).round(4)
    logit_ci_df['系数'] = np.exp(logit_ci_df['系数']).round(4)

    # 定义一个函数来处理逻辑回归的结果
    def deal_logit_result(data):

        # 这个data就是逻辑回归的结果
        for i in range(data.shape[0]):

            or_value = data.loc[i, 'means']
            lower = data.loc[i, 'lower']
            upper = data.loc[i, 'upper']
            p_value = data.loc[i, 'p值']

            data.loc[i, 'final_result'] = f'（OR={or_value}, 95%CI [{lower}, {upper}], p={p_value}）'

        return data

    logit_ci_df = deal_logit_result(logit_ci_df)
    logit_ci_df.to_csv('dataset/logit_result.csv', index=False)
    logit_ci_df.drop('final_result', axis=1, inplace=True)

    # 接下来还需要在这个函数中做可视化分析
    def logistic_visualize(logit_result):

        logit_result = copy.deepcopy(logit_result)
        min_confi = logit_result['lower'].min()
        max_confi = logit_result['upper'].max()

        logit_result.loc[logit_result['因变量'] == 'adolescent MDD', '颜色'] = '#A3C1AD'
        logit_result.loc[logit_result['因变量'] == 'adolescent UDD', '颜色'] = '#D4B9B0'
        logit_result.loc[logit_result['因变量'] == 'adolescent MDS', '颜色'] = '#6C8EBF'

        logit_result['因变量'] = logit_result['因变量'].str.replace('adolescent MDD', '青少年MDD', regex=False)
        logit_result['因变量'] = logit_result['因变量'].str.replace('adolescent UDD', '青少年UDD', regex=False)
        logit_result['因变量'] = logit_result['因变量'].str.replace('adolescent MDS', '青少年MDS', regex=False)
        logit_result['自变量'] = logit_result['自变量'].str.replace('mother depression', '母亲抑郁史', regex=False)
        logit_result['自变量'] = logit_result['自变量'].str.replace('father depression', '父亲抑郁史', regex=False)

        zibianliang = ['母亲抑郁史', '父亲抑郁史']

        # 设定逻辑回归的图片中字体大小
        fontsize = 12

        # 设定全局参数：抗锯齿、字体
        plt.rcParams['lines.antialiased'] = True
        plt.rcParams['text.antialiased'] = True
        plt.rcParams['font.family'] = 'SimSun'

        # 指定图片大小和清晰度
        plt.figure(figsize=(8, 4), dpi=1000)

        # 两个坐标轴的显示范围
        plt.xlim(0, 7.4)
        plt.ylim(min_confi - 0.2, max_confi + 0.3)

        # 刻度，标度
        plt.text(-0.25, max_confi * 1.115, 'ORs',
                 fontsize=fontsize, ha='center', va='center', rotation=0)
        plt.text(7.45, 1, '不显著',
                 fontsize=fontsize, ha='left', va='center', rotation=0)
        plt.tick_params(axis='y', labelsize=13)
        plt.xticks([])

        # 显著性线，碰到这条线就不显著
        plt.axhline(y=1, color='#DDA0DD', linestyle=(0, (1, 1)), linewidth=2.5)

        # 组间距离参数，等于五代表着组间距离为4-3=1
        zujian = 4

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.4)
        ax.spines['bottom'].set_linewidth(1.4)
        ax.tick_params(axis='y', width=1.4)

        for i, var in enumerate(zibianliang):
            logit_sub_df = logit_result.loc[logit_result['自变量'] == var, :]
            plt.text(i * zujian + 2, 0.1, var, fontsize=fontsize, ha='center', va='center', rotation=0)
            for j in range(logit_sub_df.shape[0]):
                if i == 0:
                    plt.vlines(x=(j + 1) + (i * zujian), color=logit_sub_df.iloc[j, 7], label=logit_sub_df.iloc[j, 4],
                               ymin=logit_sub_df.iloc[j, 0], ymax=logit_sub_df.iloc[j, 1], linewidth=2.5)
                    plt.scatter((j + 1) + (i * zujian), logit_sub_df.iloc[j, 6], color=logit_sub_df.iloc[j, 7], s=50)
                    if logit_sub_df.iloc[j, 5] <= 0.05:
                        plt.text((j + 1) + (i * zujian) + 0.21, logit_sub_df.iloc[j, 1] - 0.19, '*', fontsize=fontsize,
                                 ha='center', va='center', fontweight='bold')
                else:
                    plt.vlines(x=(j + 1) + (i * zujian), color=logit_sub_df.iloc[j, 7],
                               ymin=logit_sub_df.iloc[j, 0], ymax=logit_sub_df.iloc[j, 1], linewidth=2.5)
                    plt.scatter((j + 1) + (i * zujian), logit_sub_df.iloc[j, 6], color=logit_sub_df.iloc[j, 7], s=50)
                    if logit_sub_df.iloc[j, 5] <= 0.05:
                        plt.text((j + 1) + (i * zujian) + 0.21, logit_sub_df.iloc[j, 1] - 0.19, '*', fontsize=fontsize,
                                 ha='center', va='center', fontweight='bold')

        plt.legend(fontsize=fontsize, loc='best', frameon=False)
        ax.annotate('', xy=(0, 1.05), xycoords='axes fraction', xytext=(0, 0.993),
                    arrowprops=dict(arrowstyle='wedge', color='black', lw=1.2))

        ax.annotate('', xy=(1.03, 0), xycoords='axes fraction', xytext=(0.995, 0),
                    arrowprops=dict(arrowstyle='wedge', color='black', lw=1.2))
        # 画图函数的最后一步，将可视化的图片保存起来
        plt.savefig('pictures/figure_logistic_result.png', dpi=1000, bbox_inches='tight')

    logistic_visualize(logit_ci_df)






