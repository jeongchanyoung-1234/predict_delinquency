import warnings
warnings.filterwarnings('ignore')
import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder


class CreditDataset(Dataset):

    def __init__(self, x):
        self.x = x
        super().__init__()

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, idx):
        x = self.x[idx]

        return x


def get_data(config, base='c:/Users/JCY/Dacon/shinhan/') :
    train_df = pd.read_csv(os.path.join(base, 'data/train.csv'))
    test_df = pd.read_csv(os.path.join(base, 'data/test.csv'))
    train_df_target_removed = train_df.drop(columns=['credit'])
    train_df_target = train_df.iloc[:, -1]

    df = pd.concat([train_df_target_removed, test_df], axis=0)

    def birth2age(x) :
        return x * (-1) / 365

    df['age'] = df['DAYS_BIRTH'].apply(birth2age)
    df['skill'] = df['DAYS_EMPLOYED'].apply(birth2age)
    df['month'] = df['begin_month'].apply(lambda x : x * (-1))

    df['income_c'] = ''
    df.loc[df['income_total'] <= 1.575e6, 'income_c'] = 'first'
    df.loc[df['income_total'] <= 2.25e5, 'income_c'] = 'second'
    df.loc[df['income_total'] <= 1.575e5, 'income_c'] = 'third'
    df.loc[df['income_total'] <= 1.215e5, 'income_c'] = 'fourth'

    df['genderNincome'] = ''
    df.loc[(df['income_c'] == 'first') & (df['gender'] == 'F'), 'genderNincome'] = 'firstF'
    df.loc[(df['income_c'] == 'second') & (df['gender'] == 'F'), 'genderNincome'] = 'secondF'
    df.loc[(df['income_c'] == 'third') & (df['gender'] == 'F'), 'genderNincome'] = 'thirdF'
    df.loc[(df['income_c'] == 'fourth') & (df['gender'] == 'F'), 'genderNincome'] = 'fourthF'

    df.loc[(df['income_c'] == 'first') & (df['gender'] == 'M'), 'genderNincome'] = 'firstM'
    df.loc[(df['income_c'] == 'second') & (df['gender'] == 'M'), 'genderNincome'] = 'secondM'
    df.loc[(df['income_c'] == 'third') & (df['gender'] == 'M'), 'genderNincome'] = 'thirdM'
    df.loc[(df['income_c'] == 'fourth') & (df['gender'] == 'M'), 'genderNincome'] = 'fourthM'

    if config.imp :
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Commercial associate') & \
               (df['gender'] == 'F'), 'occyp_type'] = 'Sales staff'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Commercial associate') & \
               (df['gender'] == 'F') & \
               (df['income_c'] == 'first'), 'occyp_type'] = 'Managers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Commercial associate') & \
               (df['gender'] == 'F') & \
               (df['income_c'] == 'third') & \
               (df['work_phone'] == 1), 'occyp_type'] = 'Core staff'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Commercial associate') & \
               (df['gender'] == 'M'), 'occyp_type'] = 'Drivers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Commercial associate') & \
               (df['gender'] == 'M') & \
               (df['income_c'] == 'first'), 'occyp_type'] = 'Managers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Commercial associate') & \
               (df['gender'] == 'M') & \
               (df['income_c'] == 'fourth') & \
               (df['work_phone'] == 0), 'occyp_type'] = 'Laborers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Commercial associate') & \
               (df['gender'] == 'M') & \
               (df['income_c'] == 'fourth') & \
               (df['work_phone'] == 1), 'occyp_type'] = 'Drivers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Pensioner') & \
               (df['gender'] == 'F') & \
               (df['income_c'] == 'first') & \
               (df['work_phone'] == 0), 'occyp_type'] = 'Core staff'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'State servant'), 'occyp_type'] = 'Core staff'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'State servant') & \
               (df['gender'] == 'F') & \
               (df['income_c'] == 'first') & \
               (df['work_phone'] == 1), 'occyp_type'] = 'Managers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'State servant') & \
               (df['gender'] == 'F') & \
               (df['income_c'] == 'second') & \
               (df['work_phone'] == 0), 'occyp_type'] = 'Medicine staff'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'State servant') & \
               (df['gender'] == 'M') & \
               (df['income_c'] == 'first') & \
               (df['work_phone'] == 1), 'occyp_type'] = 'High skill tech staff'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'State servant') & \
               (df['gender'] == 'M') & \
               (df['income_c'] == 'fouth') & \
               (df['work_phone'] == 0), 'occyp_type'] = 'Laborers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'State servant') & \
               (df['gender'] == 'M') & \
               (df['income_c'] == 'third') & \
               (df['work_phone'] == 1), 'occyp_type'] = 'Drivers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Working'), 'occyp_type'] = 'Laborers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Working') & \
               (df['gender'] == 'F'), 'occyp_type'] = 'Sales staff'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'State servant') & \
               (df['gender'] == 'F') & \
               (df['income_c'] == 'first') & \
               (df['work_phone'] == 1), 'occyp_type'] = 'Managers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'State servant') & \
               (df['gender'] == 'F') & \
               (df['income_c'] == 'second') & \
               (df['work_phone'] == 1), 'occyp_type'] = 'Core staff'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Student') & \
               (df['income_c'] == 'fourth'), 'occyp_type'] = 'Laborers'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Student') & \
               (df['income_c'] == 'second'), 'occyp_type'] = 'Core staff'
        df.loc[df['occyp_type'].isnull() & \
               (df['income_type'] == 'Pensioner'), 'occyp_type'] = 'Laborers'
    else :
        df.drop(columns=['occyp_type'], inplace=True)

    df['DAYS_BIRTH_month'] = np.floor((-df['DAYS_BIRTH']) / 30) - (
                (np.floor((-df['DAYS_BIRTH']) / 30) / 12).astype(int) * 12)
    df['DAYS_BIRTH_week'] = np.floor((-df['DAYS_BIRTH']) / 7) - (
                (np.floor((-df['DAYS_BIRTH']) / 7) / 4).astype(int) * 4)

    # DAYS_EMPLOYED
    df['DAYS_EMPLOYED_month'] = np.floor((-df['DAYS_EMPLOYED']) / 30) - (
                (np.floor((-df['DAYS_EMPLOYED']) / 30) / 12).astype(int) * 12)
    df['DAYS_EMPLOYED_week'] = np.floor((-df['DAYS_EMPLOYED']) / 7) - (
                (np.floor((-df['DAYS_EMPLOYED']) / 7) / 4).astype(int) * 4)

    # before_EMPLOYED
    df['before_EMPLOYED'] = df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']
    df['before_EMPLOYED_month'] = np.floor((-df['before_EMPLOYED']) / 30) - (
                (np.floor((-df['before_EMPLOYED']) / 30) / 12).astype(int) * 12)
    df['before_EMPLOYED_week'] = np.floor((-df['before_EMPLOYED']) / 7) - (
                (np.floor((-df['before_EMPLOYED']) / 7) / 4).astype(int) * 4)

    df.drop(columns=['begin_month', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'income_c'], inplace=True)

    object_col = []
    time_columns = ['DAYS_BIRTH_month', 'DAYS_BIRTH_week', 'DAYS_EMPLOYED_month', 'DAYS_EMPLOYED_week',
                    'before_EMPLOYED_month', 'before_EMPLOYED_week']
    for col in df.columns :
        if df[col].dtype == 'object' :
            object_col.append(col)
    object_col += ['FLAG_MOBIL', 'work_phone', 'phone', 'email']
    object_col += time_columns
    enc = OneHotEncoder()
    enc.fit(df.loc[:, object_col][:26457]) # train set만 encoding에 사용

    onehot_df = pd.DataFrame(enc.transform(df.loc[:, object_col]).toarray(),
                             columns=enc.get_feature_names(object_col))

    new_df = pd.concat([df.reset_index(drop=True), onehot_df], axis=1)
    new_df.drop(columns=object_col, inplace=True)

    train_df = new_df[:26457]
    train_df = pd.concat([train_df, train_df_target], axis=1)
    test_df = new_df[26457 :]

    if config.objective == 'ae':
        train = train_df.drop(columns=['index'])

        # 훈련은 train만으로, 변환은 전체 데이터를 인코딩
        X = train[enc.get_feature_names(object_col)].values
        whole_data = new_df[enc.get_feature_names(object_col)].values

        print(X.shape)
        return X, whole_data

    if config.objective == 'clf':
        train = train_df.drop(columns=['index'])
        test = test_df.drop(columns=['index'])
        enc_list = enc.get_feature_names(object_col)


        enc_result = pd.read_csv('./data/encode_result_gni.csv')
        train = pd.concat([train.drop(columns=enc_list).reset_index(drop=True),
                           enc_result[:26457].reset_index(drop=True)],
                          axis=1).reset_index(drop=True)
        test = pd.concat([test.drop(columns=enc_list).reset_index(drop=True),
                          enc_result[26457 :].reset_index(drop=True)],
                         axis=1).reset_index(drop=True)

        print('{} features are reduced to {}-d'.format(len(enc.get_feature_names(object_col)),
                                                           len(enc_result.columns)))
        print('Shape: train {} test {} total {}'.format(train.shape, test.shape, new_df.shape))

        return train, train_df_target, test, new_df, enc_list


def get_loaders(config, x):
    train_loader = DataLoader(
        dataset=CreditDataset(x),
        batch_size=config.batch_size,
        shuffle=True,
    )

    return train_loader