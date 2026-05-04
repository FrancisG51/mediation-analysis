import pandas as pd


def have_data_all_function():
    brain_data = pd.read_csv('dataset/brain_all.csv')
    other_data = pd.read_csv('dataset/miced.csv')
    data_all = pd.merge(other_data, brain_data, how='inner', on='src_subject_id')
    data_all.reset_index(drop=True, inplace=True)
    print('所有数据的形状：', data_all.shape)

    rename_lie = {'fam_history_q6a_depression': 'father_depression',
                  'fam_history_q6d_depression': 'mother_depression'}
    data_all = data_all.rename(columns=rename_lie)

    rename_lie = {'adolescent_MDD': 'adolescent MDD',
                  'adolescent_UDD': 'adolescent UDD',
                  'adolescent_MDS': 'adolescent MDS',
                  'father_depression': 'father depression',
                  'mother_depression': 'mother depression'}
    data_all = data_all.rename(columns=rename_lie)

    data_all.to_csv('dataset/data_all.csv', index=False)

