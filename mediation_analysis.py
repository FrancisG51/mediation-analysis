import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt


def mediation_analysis():

    logistic_result = pd.read_csv('dataset/logit_result.csv')
    logistic_result = logistic_result.loc[logistic_result['p值'] <= 0.05, :]
    logistic_result.reset_index(drop=True, inplace=True)

    yinbianliang = ['adolescent MDD', 'adolescent UDD', 'adolescent MDS']
    zibianliang = ['father depression', 'mother depression']
    xiebianliang = ['demo_prnt_ed_v2', 'demo_brthdat_v2', 'demo_comb_income_v2',
                    'demo_race_a_p___10', 'demo_race_a_p___11', 'male']

    data_all = pd.read_csv('dataset/data_all.csv')
    zhongjiebianliang = data_all.columns.to_list()
    zhongjiebianliang.remove('src_subject_id')
    zhongjiebianliang = list(set(zhongjiebianliang) - set(zibianliang) - set(xiebianliang) - set(yinbianliang))
    print('中介变量的数量：', len(zhongjiebianliang))

    def mediation_analysis(X, Y, M):
        # 在函数内部创建一个dataframe用来进行变量赋值
        column_names = ['自变量', '因变量', '中介变量',
                        '（单一回归，自变量到因变量）系数（总效应）', '（单一回归，自变量到因变量）P值',
                        '（单一回归，自变量到因变量）置信下限', '（单一回归，自变量到因变量）置信上限',
                        '（单一回归，自变量到中介变量）系数', '（单一回归，自变量到中介变量）P值',
                        '（单一回归，自变量到中介变量）置信下限', '（单一回归，自变量到中介变量）置信上限',
                        '（一起回归）自变量到因变量，系数（直接效应）', '（一起回归）自变量到因变量，P值',
                        '（一起回归）自变量到因变量，置信下限', '（一起回归）自变量到因变量，置信上限',
                        '（一起回归）中介变量到因变量，系数', '（一起回归）中介变量到因变量，P值',
                        '（一起回归）中介变量到因变量，置信下限', '（一起回归）中介变量到因变量，置信上限',
                        '中介效应值（间接效应）']
        mediation_df = pd.DataFrame(columns=column_names)
        nan_hang = [np.nan] * len(column_names)
        mediation_df.loc[0] = nan_hang

        # 开始赋值XYM
        mediation_df.loc[0, '自变量'] = X
        mediation_df.loc[0, '因变量'] = Y
        mediation_df.loc[0, '中介变量'] = M

        # 先做总效应，自变量到因变量的单一逻辑回归
        zibianliang_num1 = zibianliang + xiebianliang
        zibianliang_num1 = data_all[zibianliang_num1]
        zibianliang_num1 = sm.add_constant(zibianliang_num1)
        yinbianliang_num1 = data_all[Y]
        # 获取了数据之后开始做逻辑回归
        logit_mod = sm.Logit(yinbianliang_num1, zibianliang_num1).fit(maxiter=20, disp=False)
        mediation_df.loc[0, '（单一回归，自变量到因变量）系数（总效应）'] = logit_mod.params[X]
        mediation_df.loc[0, '（单一回归，自变量到因变量）P值'] = logit_mod.pvalues[X]
        logit_mod_conf_df = logit_mod.conf_int()
        logit_mod_X_conf_lower = logit_mod_conf_df.loc[logit_mod_conf_df.index == X, 0].values[0]
        logit_mod_X_conf_upper = logit_mod_conf_df.loc[logit_mod_conf_df.index == X, 1].values[0]
        mediation_df.loc[0, '（单一回归，自变量到因变量）置信下限'] = logit_mod_X_conf_lower
        mediation_df.loc[0, '（单一回归，自变量到因变量）置信上限'] = logit_mod_X_conf_upper

        # 开始做自变量到中介变量的线性回归
        zibianliang_num2 = zibianliang + xiebianliang
        zibianliang_num2 = data_all[zibianliang_num2]
        zibianliang_num2 = sm.add_constant(zibianliang_num2)
        yinbianliang_num2 = data_all[M]
        # 获取了数据之后开始做线性回归
        ols_mod = sm.OLS(yinbianliang_num2, zibianliang_num2).fit(maxiter=20, disp=False)
        mediation_df.loc[0, '（单一回归，自变量到中介变量）系数'] = ols_mod.params[X]
        mediation_df.loc[0, '（单一回归，自变量到中介变量）P值'] = ols_mod.pvalues[X]
        # 开始找置信区间95上下限
        ols_mod_conf_df = ols_mod.conf_int()
        ols_mod_conf_lower = ols_mod_conf_df.loc[ols_mod_conf_df.index == X, 0].values[0]
        ols_mod_conf_upper = ols_mod_conf_df.loc[ols_mod_conf_df.index == X, 1].values[0]
        mediation_df.loc[0, '（单一回归，自变量到中介变量）置信下限'] = ols_mod_conf_lower
        mediation_df.loc[0, '（单一回归，自变量到中介变量）置信上限'] = ols_mod_conf_upper

        # 开始做一起回归，自变量加协变量加中介变量对于因变量的逻辑回归
        zibianliang_num3 = zibianliang + xiebianliang
        zibianliang_num3.append(M)
        zibianliang_num3 = data_all[zibianliang_num3]
        zibianliang_num3 = sm.add_constant(zibianliang_num3)
        yinbianliang_num3 = data_all[Y]
        # 获取了数据之后开始做逻辑回归
        logit_mod = sm.Logit(yinbianliang_num3, zibianliang_num3).fit(maxiter=20, disp=False)
        mediation_df.loc[0, '（一起回归）自变量到因变量，系数（直接效应）'] = logit_mod.params[X]
        mediation_df.loc[0, '（一起回归）自变量到因变量，P值'] = logit_mod.pvalues[X]
        mediation_df.loc[0, '（一起回归）中介变量到因变量，系数'] = logit_mod.params[M]
        mediation_df.loc[0, '（一起回归）中介变量到因变量，P值'] = logit_mod.pvalues[M]
        # 开始赋值百分之95的上下限
        logit_mod_conf_df = logit_mod.conf_int()
        logit_mod_M_conf_lower = logit_mod_conf_df.loc[logit_mod_conf_df.index == M, 0].values[0]
        logit_mod_M_conf_upper = logit_mod_conf_df.loc[logit_mod_conf_df.index == M, 1].values[0]
        mediation_df.loc[0, '（一起回归）中介变量到因变量，置信下限'] = logit_mod_M_conf_lower
        mediation_df.loc[0, '（一起回归）中介变量到因变量，置信上限'] = logit_mod_M_conf_upper
        logit_mod_X_conf_lower = logit_mod_conf_df.loc[logit_mod_conf_df.index == X, 0].values[0]
        logit_mod_X_conf_upper = logit_mod_conf_df.loc[logit_mod_conf_df.index == X, 1].values[0]
        mediation_df.loc[0, '（一起回归）自变量到因变量，置信下限'] = logit_mod_X_conf_lower
        mediation_df.loc[0, '（一起回归）自变量到因变量，置信上限'] = logit_mod_X_conf_upper

        # 对于中介效应值的赋值
        ab = (ols_mod.params[X] * logit_mod.params[M])
        mediation_df.loc[0, '中介效应值（间接效应）'] = ab

        return mediation_df

    # 创建空的列表用来表格的拼接
    column_names = ['自变量', '因变量', '中介变量',
                    '（单一回归，自变量到因变量）系数（总效应）', '（单一回归，自变量到因变量）P值',
                    '（单一回归，自变量到因变量）置信下限', '（单一回归，自变量到因变量）置信上限',
                    '（单一回归，自变量到中介变量）系数', '（单一回归，自变量到中介变量）P值',
                    '（单一回归，自变量到中介变量）置信下限', '（单一回归，自变量到中介变量）置信上限',
                    '（一起回归）自变量到因变量，系数（直接效应）', '（一起回归）自变量到因变量，P值',
                    '（一起回归）自变量到因变量，置信下限', '（一起回归）自变量到因变量，置信上限',
                    '（一起回归）中介变量到因变量，系数', '（一起回归）中介变量到因变量，P值',
                    '（一起回归）中介变量到因变量，置信下限', '（一起回归）中介变量到因变量，置信上限',
                    '中介效应值（间接效应）']
    concat_df = pd.DataFrame(columns=column_names)
    nan_hang = [np.nan] * len(column_names)
    concat_df.loc[0] = nan_hang

    for m in tqdm(zhongjiebianliang):
        for i in range(logistic_result.shape[0]):
            x = logistic_result.loc[i, '自变量']
            y = logistic_result.loc[i, '因变量']
            concat_df = pd.concat([concat_df, mediation_analysis(x, y, m)])

    print('刚运算完的dataframe形状：', concat_df.shape)
    concat_df = concat_df.dropna(thresh=int(1 * concat_df.shape[1]), axis=0)
    concat_df.reset_index(drop=True, inplace=True)
    print('消除空缺行后，完整的原始表格：', concat_df.shape)

    # 画图gwas的-lgp值的图
    def gwas_visualize_function(mediation_result):

        # 提取基期的脑测量数据
        def have_baseline(x):
            x = x.loc[x['eventname'] == 'baseline_year_1_arm_1', :]
            x = x.drop('eventname', axis=1)
            x.reset_index(drop=True, inplace=True)
            return x

        mri_y_smr_thk_dst = pd.read_csv('raw_data/mri_y_smr_thk_dst.csv')
        mri_y_smr_thk_dst = have_baseline(mri_y_smr_thk_dst)
        mri_y_smr_sulc_dst = pd.read_csv('raw_data/mri_y_smr_sulc_dst.csv')
        mri_y_smr_sulc_dst = have_baseline(mri_y_smr_sulc_dst)
        mri_y_smr_area_dst = pd.read_csv('raw_data/mri_y_smr_area_dst.csv')
        mri_y_smr_area_dst = have_baseline(mri_y_smr_area_dst)
        mri_y_smr_t1_gray_dst = pd.read_csv('raw_data/mri_y_smr_t1_gray_dst.csv')
        mri_y_smr_t1_gray_dst = have_baseline(mri_y_smr_t1_gray_dst)
        mri_y_smr_t1_contr_dst = pd.read_csv('raw_data/mri_y_smr_t1_contr_dst.csv')
        mri_y_smr_t1_contr_dst = have_baseline(mri_y_smr_t1_contr_dst)
        mri_y_smr_t1_white_dst = pd.read_csv('raw_data/mri_y_smr_t1_white_dst.csv')
        mri_y_smr_t1_white_dst = have_baseline(mri_y_smr_t1_white_dst)
        mri_y_smr_vol_dst = pd.read_csv('raw_data/mri_y_smr_vol_dst.csv')
        mri_y_smr_vol_dst = have_baseline(mri_y_smr_vol_dst)
        mri_y_smr_t1_aseg = pd.read_csv('raw_data/mri_y_smr_t1_aseg.csv')
        mri_y_smr_t1_aseg = have_baseline(mri_y_smr_t1_aseg)
        mri_y_smr_vol_aseg = pd.read_csv('raw_data/mri_y_smr_vol_aseg.csv')
        mri_y_smr_vol_aseg = have_baseline(mri_y_smr_vol_aseg)

        dataframe_list = [mri_y_smr_thk_dst, mri_y_smr_sulc_dst, mri_y_smr_area_dst, mri_y_smr_vol_dst,
                          mri_y_smr_t1_white_dst, mri_y_smr_t1_gray_dst, mri_y_smr_t1_contr_dst,
                          mri_y_smr_vol_aseg, mri_y_smr_t1_aseg]

        column_names = ['中介变量', '脑测量的表格名称', '表格名称的含义']
        gwas_concat = pd.DataFrame(columns=column_names)
        nan_hang = [np.nan] * len(column_names)
        gwas_concat.loc[0, :] = nan_hang

        def var_name(var, all_var=locals()):
            return [var_name for var_name in all_var if all_var[var_name] is var][0]

        def get_columnname(df):
            columns_list = df.columns.to_list()
            columns_list.remove('src_subject_id')
            linshi_df = pd.DataFrame({'中介变量': columns_list})
            return linshi_df

        for i, df in enumerate(dataframe_list):
            df_linshi = get_columnname(df)
            df_linshi['脑测量的表格名称'] = var_name(df, all_var=locals())
            gwas_concat = pd.concat([gwas_concat, df_linshi])

        # 给对应的表格名称相应的含义
        gwas_concat.loc[gwas_concat.loc[:, '脑测量的表格名称'] == 'mri_y_smr_thk_dst', '表格名称的含义'] = 'Cortical Thickness'
        gwas_concat.loc[gwas_concat.loc[:, '脑测量的表格名称'] == 'mri_y_smr_sulc_dst', '表格名称的含义'] = 'Sulcus Depth'
        gwas_concat.loc[gwas_concat.loc[:, '脑测量的表格名称'] == 'mri_y_smr_area_dst', '表格名称的含义'] = 'Cortical Area'
        gwas_concat.loc[gwas_concat.loc[:, '脑测量的表格名称'] == 'mri_y_smr_vol_dst', '表格名称的含义'] = 'Brain Volume'
        gwas_concat.loc[gwas_concat.loc[:, '脑测量的表格名称'] == 'mri_y_smr_t1_white_dst', '表格名称的含义'] = 'White Matter'
        gwas_concat.loc[gwas_concat.loc[:, '脑测量的表格名称'] == 'mri_y_smr_t1_gray_dst', '表格名称的含义'] = 'Gray Matter'
        gwas_concat.loc[gwas_concat.loc[:, '脑测量的表格名称'] == 'mri_y_smr_t1_contr_dst', '表格名称的含义'] = 'Contrast'
        gwas_concat.loc[gwas_concat.loc[:, '脑测量的表格名称'] == 'mri_y_smr_vol_aseg', '表格名称的含义'] = 'Volume Aseg'
        gwas_concat.loc[gwas_concat.loc[:, '脑测量的表格名称'] == 'mri_y_smr_t1_aseg', '表格名称的含义'] = 'T1 Aseg'

        gwas_concat = gwas_concat.dropna(thresh=int(1 * gwas_concat.shape[1]), axis=0)
        gwas_concat.reset_index(drop=True, inplace=True)
        gwas_concat.to_csv('dataset/brain_dictionary.csv', index=False)

        # 看gwas_concat的表格信息
        print('表格gwas_concat的形状', gwas_concat.shape)
        print('在gwas中有多少个脑测量变量', len(set(gwas_concat.loc[:, '中介变量'].to_list())))

        match_df = pd.DataFrame({
            '变量名称': ['father depression', 'mother depression', 'adolescent MDD', 'adolescent UDD', 'adolescent MDS'],
            '变量类型': ['自变量', '自变量', '因变量', '因变量', '因变量'],
            'p值类型': ['（单一回归，自变量到中介变量）P值', '（单一回归，自变量到中介变量）P值',
                       '（一起回归）中介变量到因变量，P值', '（一起回归）中介变量到因变量，P值', '（一起回归）中介变量到因变量，P值']})

        fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True, dpi=1000)

        fontsize = 25
        # 统一的y轴标签，-log10(p)
        fig.text(x=0.031, y=0.5, s='-lg(p) of Depression-Structural Correlation Significance',
                 rotation=90, va='center', ha='center', fontsize=fontsize)

        for i in range(match_df.shape[0]):

            var_name = match_df.loc[i, '变量名称']
            var_leixing = match_df.loc[i, '变量类型']
            p_leixing = match_df.loc[i, 'p值类型']
            title_name = match_df.loc[i, '变量名称']

            # 利用match_df从concat_df中提取数据
            linshi_df = mediation_result.loc[mediation_result[var_leixing] == var_name, :]
            linshi_df.reset_index(drop=True, inplace=True)
            linshi_df.drop_duplicates(subset='中介变量', keep='first', inplace=True)    # 对于指定的一列去重，保留重复的第一行
            linshi_df.reset_index(drop=True, inplace=True)

            linshi_df = linshi_df[[var_leixing, '中介变量', p_leixing]]
            linshi_df['-log10(p)'] = -np.log10(linshi_df[p_leixing])
            linshi_df = pd.merge(gwas_concat, linshi_df, how='inner', on='中介变量')
            linshi_df.reset_index(drop=True, inplace=True)

            axs[i].set_title(title_name, fontsize=fontsize, x=0, y=1.025, ha='left', va='bottom')
            if i == 0:
                axs[i].text(x=1193, y=-np.log10(0.05), s='p=0.05', rotation=0, va='center', ha='center',
                            fontsize=fontsize - 4)

            axs[i].axhline(y=-np.log10(0.05), color='#2F4F4F', linestyle='--', alpha=0.8)
            axs[i].set_xlim([-6, 1140])
            axs[i].set_ylim([-0.01, linshi_df['-log10(p)'].max() * 1.05])
            axs[i].set_xticks([])

            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['left'].set_linewidth(1.5)
            axs[i].spines['bottom'].set_linewidth(1.5)
            axs[i].tick_params(axis='x', width=1.5)
            axs[i].tick_params(axis='y', width=1.5, labelsize=fontsize - 4)

            axs[i].annotate('', xy=(0, 1.04), xycoords='axes fraction', xytext=(0, 0.993),
                            arrowprops=dict(arrowstyle='wedge', color='black', lw=1.5))
            axs[i].annotate('', xy=(1.013, 0), xycoords='axes fraction', xytext=(0.999, 0),
                            arrowprops=dict(arrowstyle='wedge', color='black', lw=1.5))
            for j in range(linshi_df.shape[0]):
                if j in {1130, 1}:
                    if linshi_df.loc[j, '-log10(p)'] >= -np.log10(0.05):
                        axs[i].vlines(x=j, color='#FFB347', ymin=0, ymax=linshi_df.loc[j, '-log10(p)'],
                                      label='significant', alpha=1)
                    else:
                        axs[i].vlines(x=j, color='#6C8EBF', ymin=0, ymax=linshi_df.loc[j, '-log10(p)'],
                                      label='not significant', alpha=1)
                else:
                    if linshi_df.loc[j, '-log10(p)'] >= -np.log10(0.05):
                        axs[i].vlines(x=j, color='#FFB347', ymin=0, ymax=linshi_df.loc[j, '-log10(p)'], alpha=1)
                    else:
                        axs[i].vlines(x=j, color='#6C8EBF', ymin=0, ymax=linshi_df.loc[j, '-log10(p)'], alpha=1)

        axs[0].legend(fontsize=fontsize, frameon=False, bbox_to_anchor=(1, 1.38),
                      loc='upper right', bbox_transform=axs[0].transAxes)

        label_change_onrow = [0]
        for k in range(linshi_df.shape[0] - 1):
            on_label = linshi_df.loc[k, '脑测量的表格名称']
            under_label = linshi_df.loc[k + 1, '脑测量的表格名称']
            if on_label != under_label:
                label_change_onrow.append(k)
        label_change_onrow.append(1140)

        text_label = []
        for i in range(len(label_change_onrow) - 1):
            start = label_change_onrow[i]
            end = label_change_onrow[i + 1]
            mid = int((start + end) / 2)
            text_label.append(mid)
        text_label.remove(1119)
        text_label.append(1134)

        axs[-1].set_xticks([150, 301, 452, 603, 754, 905, 1056, 1098])
        axs[-1].set_xticklabels([])

        # 统一设定显示变量标签的text函数内的各个参数
        y_value = -0.05
        ha_value = 'right'
        va_value = 'top'
        rotation_value = 45

        for i in text_label:
            label_name = linshi_df.loc[i, '表格名称的含义']
            axs[-1].text(i + 10, y_value, label_name, ha=ha_value, va=va_value, fontsize=fontsize - 4,
                         rotation=rotation_value)

        # 一切可视化都画完了，最后将gwas的图保存起来
        plt.savefig('pictures/gwas_visualize.png', dpi=1000, bbox_inches='tight')

    # 跳出函数
    # 在这一步进行gwas画图
    gwas_visualize_function(concat_df)

    concat_df = concat_df.round(4)
    concat_df.to_csv('dataset/mediation_result_without_pvalue.csv', index=False)
    # gwas可视化完成，然后筛选显著的中介效应，保存结果
    concat_df = concat_df.loc[concat_df['（单一回归，自变量到因变量）P值'] <= 0.05, :]
    concat_df = concat_df.loc[concat_df['（单一回归，自变量到中介变量）P值'] <= 0.05, :]
    concat_df = concat_df.loc[concat_df['（一起回归）中介变量到因变量，P值'] <= 0.05, :]
    concat_df.reset_index(drop=True, inplace=True)
    print('显著性筛选之后的表格形状：', concat_df.shape)
    concat_df = concat_df.round(4)
    concat_df.to_csv('dataset/mediation_result.csv', index=False)

