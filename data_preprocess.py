import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings('ignore')


def process_raw_data_no_brain():

    def processdf_baseline(list_name, dataframe, var_char):
        dataframe = dataframe[list_name]
        dataframe = dataframe.loc[dataframe['eventname'] == 'baseline_year_1_arm_1']
        dataframe = dataframe.drop('eventname', axis=1)
        dataframe.replace([888, 777, 999, 555, 666, 444, 222, 333], np.nan, inplace=True)
        dataframe.reset_index(drop=True, inplace=True)

        dataframe[var_char] = dataframe.apply(lambda row: 1 if any(row == 1) else 0, axis=1)
        dataframe = dataframe[['src_subject_id', var_char]]
        return dataframe

    # 青少年抑郁数据
    youth_depress = pd.read_csv('raw_data/mh_p_ksads_ss.csv')
    DBD_list = ['src_subject_id', 'eventname', 'ksads_2_836_p', 'ksads_2_831_p', 'ksads_2_834_p',
                'ksads2_2_799_p', 'ksads2_2_931_p', 'ksads2_2_936_p', 'ksads2_2_937_p']
    MDD_list = ['src_subject_id', 'eventname', 'ksads_1_840_p', 'ksads_1_841_p', 'ksads_1_842_p',
                'ksads2_1_790_p', 'ksads2_1_791_p', 'ksads2_1_792_p']
    PDD_list = ['src_subject_id', 'eventname', 'ksads_1_843_p', 'ksads_1_845_p', 'ksads_1_844_p',
                'ksads2_1_793_p', 'ksads2_1_794_p', 'ksads2_1_795_p']
    UDD_list = ['src_subject_id', 'eventname', 'ksads_1_847_p', 'ksads_1_846_p',
                'ksads2_1_796_p']
    MDS_list = ['src_subject_id', 'eventname', 'ksads_1_185_p', 'ksads_1_186_p',
                'ksads2_1_176_p', 'ksads2_1_177_p']
    DBD_df = processdf_baseline(DBD_list, youth_depress, 'adolescent_DBD')
    MDD_df = processdf_baseline(MDD_list, youth_depress, 'adolescent_MDD')
    PDD_df = processdf_baseline(PDD_list, youth_depress, 'adolescent_PDD')
    UDD_df = processdf_baseline(UDD_list, youth_depress, 'adolescent_UDD')
    MDS_df = processdf_baseline(MDS_list, youth_depress, 'adolescent_MDS')

    adolescent_depress = pd.merge(DBD_df, MDD_df, how='outer', on='src_subject_id')
    adolescent_depress = pd.merge(adolescent_depress, PDD_df, how='outer', on='src_subject_id')
    adolescent_depress = pd.merge(adolescent_depress, UDD_df, how='outer', on='src_subject_id')
    adolescent_depress = pd.merge(adolescent_depress, MDS_df, how='outer', on='src_subject_id').reset_index(drop=True,
                                                                                                            inplace=False)
    drop_column = []
    for column in adolescent_depress.columns:
        one_count = (adolescent_depress[column] == 1).sum()
        if one_count <= 50:
            drop_column.append(column)
    drop_column.remove('src_subject_id')
    adolescent_depress.drop(labels=drop_column, axis=1, inplace=True)

    # 抑郁家族史
    family_history = pd.read_csv('raw_data/mh_p_fhx.csv')
    family_history = family_history[['src_subject_id', 'eventname',
                                     'fam_history_6_yes_no',
                                     'fam_history_q6a_depression', 'fam_history_q6d_depression']].reset_index(drop=True,
                                                                                                              inplace=False)
    need_imputation = ['fam_history_q6a_depression', 'fam_history_q6d_depression']
    for i in range(family_history.shape[0]):
        if family_history.loc[i, 'fam_history_6_yes_no'] == 0:
            family_history.loc[i, need_imputation] = 0
    family_history.drop('fam_history_6_yes_no', axis=1, inplace=True)
    family_history = family_history.loc[family_history['eventname'] == 'baseline_year_1_arm_1']
    family_history = family_history.drop('eventname', axis=1)
    family_history.replace([888, 777, 999, 555, 666, 444, 222, 333], np.nan, inplace=True)
    family_history.reset_index(drop=True, inplace=True)

    # 人口统计数据
    demographic = pd.read_csv('raw_data/abcd_p_demo.csv')
    demographic = demographic[['src_subject_id', 'eventname',
                               'demo_prnt_ed_v2', 'demo_brthdat_v2', 'demo_comb_income_v2',
                               'demo_race_a_p___10', 'demo_race_a_p___11', 'demo_sex_v2']]
    demographic['male'] = np.where(demographic['demo_sex_v2'] == 1, 1, 0)
    demographic = demographic.loc[demographic['eventname'] == 'baseline_year_1_arm_1']
    demographic = demographic.drop(['eventname', 'demo_sex_v2'], axis=1)
    demographic.replace([888, 777, 999, 555, 666, 444, 222, 333], np.nan, inplace=True)
    demographic.reset_index(drop=True, inplace=True)

    data_not_brain = pd.merge(adolescent_depress, family_history, on='src_subject_id', how='outer')
    data_not_brain = pd.merge(data_not_brain, demographic, on='src_subject_id', how='outer')
    data_not_brain.replace([999, 888, 777, 666, 555, 444, 333, 222], np.nan, inplace=True)
    data_not_brain.reset_index(drop=True, inplace=True)

    thresh_hang = int(0.9999 * data_not_brain.shape[1])
    data_not_brain = data_not_brain.dropna(thresh=thresh_hang, axis=0)
    data_not_brain.reset_index(drop=True, inplace=True)

    data_not_brain.to_csv('dataset/to_mice.csv', index=False)


