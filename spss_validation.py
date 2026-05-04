import pandas as pd


def spss_valid():

    data_all = pd.read_csv('dataset/data_all.csv')
    var_names = ['mrisdp_1011',
                 'adolescent MDD', 'adolescent UDD', 'adolescent MDS',
                 'father depression', 'mother depression',
                 'demo_prnt_ed_v2', 'demo_brthdat_v2', 'demo_comb_income_v2',
                 'demo_race_a_p___10', 'demo_race_a_p___11', 'male']
    data_all = data_all[var_names]

    rename_dict = {'mrisdp_1011': 'mri',
                   'adolescent MDD': 'adMDD',
                   'adolescent UDD': 'adUDD',
                   'adolescent MDS': 'adMDS',
                   'father depression': 'fa',
                   'mother depression': 'mo',
                   'demo_prnt_ed_v2': 'edu',
                   'demo_brthdat_v2': 'birth',
                   'demo_comb_income_v2': 'incom',
                   'demo_race_a_p___10': 'ra10',
                   'demo_race_a_p___11': 'ra11',
                   'male': 'male'}
    data_all.rename(columns=rename_dict, inplace=True)
    data_all.to_csv('dataset/spss_vali.csv', index=False)
