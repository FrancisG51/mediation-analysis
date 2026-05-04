import pandas as pd
from sklearn.preprocessing import StandardScaler


def have_brain_data():

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

    brain = pd.merge(mri_y_smr_thk_dst, mri_y_smr_sulc_dst, how='outer', on='src_subject_id')
    brain = pd.merge(brain, mri_y_smr_area_dst, how='outer', on='src_subject_id')
    brain = pd.merge(brain, mri_y_smr_t1_gray_dst, how='outer', on='src_subject_id')
    brain = pd.merge(brain, mri_y_smr_t1_contr_dst, how='outer', on='src_subject_id')
    brain = pd.merge(brain, mri_y_smr_t1_white_dst, how='outer', on='src_subject_id')
    brain = pd.merge(brain, mri_y_smr_vol_dst, how='outer', on='src_subject_id')
    brain = pd.merge(brain, mri_y_smr_t1_aseg, how='outer', on='src_subject_id')
    brain = pd.merge(brain, mri_y_smr_vol_aseg, how='outer', on='src_subject_id')

    thresh_lie = int(0.7 * brain.shape[0])
    brain = brain.dropna(thresh=thresh_lie, axis=1)
    thresh_hang = int(1 * brain.shape[1])
    brain = brain.dropna(thresh=thresh_hang, axis=0)
    brain.reset_index(drop=True, inplace=True)

    unique_counts = brain.nunique()
    columns_keep = unique_counts[unique_counts > 1].index
    brain = brain.loc[:, columns_keep]
    brain.reset_index(drop=True, inplace=True)

    scaler = StandardScaler()
    brain_columns = list(set(list(brain.columns)) - set(['src_subject_id']))
    brain[brain_columns] = scaler.fit_transform(brain[brain_columns])
    brain.reset_index(drop=True, inplace=True)

    brain.to_csv('dataset/brain_all.csv', index=False)

    print('脑测量变量的数量，包括id列：', len(set(brain.columns)))
    print('脑测量变量的表格形状：', brain.shape)







