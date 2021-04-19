import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class CreditDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        super().__init__()

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y


def get_data(config, base='c:/Users/JCY/Dacon/shinhan/') :
    import os
    from sklearn.preprocessing import OneHotEncoder

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


    if config.imp:
        df['income_c'] = ''
        df.loc[df['income_total'] <= 1.575e6, 'income_c'] = 'first'
        df.loc[df['income_total'] <= 2.25e5, 'income_c'] = 'second'
        df.loc[df['income_total'] <= 1.575e5, 'income_c'] = 'third'
        df.loc[df['income_total'] <= 1.215e5, 'income_c'] = 'fourth'

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
        df.fillna(value='NAN', inplace=True)

        df.drop(columns=['income_c'], inplace=True)
    else :
        df.drop(columns=['occyp_type'], inplace=True)

    object_col = []
    for col in df.columns :
        if df[col].dtype == 'object' :
            object_col.append(col)
    enc = OneHotEncoder()
    enc.fit(df.loc[:, object_col])

    onehot_df = pd.DataFrame(enc.transform(df.loc[:, object_col]).toarray(),
                             columns=enc.get_feature_names(object_col))


    if config.objective == 'ae':
        train_df = onehot_df[:26457]
        test_df = onehot_df[26457:]
        whole_df = pd.concat([train_df.reset_index(drop=True), test_df.reset_index(drop=True)], axis=0)

        train_X = train_df.values
        train_y = train_df_target.values

        return train_X, train_y, whole_df


    elif config.objective == 'normal':
        new_df = pd.concat([df.reset_index(drop=True), onehot_df], axis=1)
        new_df.drop(columns=object_col, inplace=True)

        train_df = new_df[:26457]
        test_df = new_df[26457:]

        train_X = train_df.values[:, 1:]
        train_y = train_df_target.values
        test_X = test_df.values[:, 1:]

        return train_X, train_y, test_X


def get_loaders(config, x, y):
    from sklearn.model_selection import train_test_split

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=config.train_ratio)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5)

    train_x, train_y, val_x, val_y, test_x, test_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float(), \
                                                     torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float(), \
                                                     torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float(),


    train_loader = DataLoader(
        dataset=CreditDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        dataset=CreditDataset(val_x, val_y),
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=CreditDataset(test_x, test_y),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader