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

def get_data(config, base='C:/Users/JCY/Dacon/shinhan/data') :
    import os
    from copy import deepcopy

    import numpy as np
    import pandas as pd

    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler


    def birth2age(x) :
        return x * (-1) / 365

    train_df = pd.read_csv(os.path.join(base, '../data/train.csv'))
    test_df = pd.read_csv(os.path.join(base, '../data/test.csv'))
    train_df_target_removed = train_df.drop(columns=['credit'])
    train_df_target = train_df.iloc[:, -1]

    df = pd.concat([train_df_target_removed, test_df], axis=0)
    print('Shape: train {} test {} total {}'.format(train_df.shape, test_df.shape, df.shape))


    df['age'] = df['DAYS_BIRTH'].apply(birth2age)
    df['skill'] = df['DAYS_EMPLOYED'].apply(birth2age)
    df['month'] = df['begin_month'].apply(lambda x : x * (-1))

    df['income_c'] = ''
    df.loc[df['income_total'] <= 1.575e6, 'income_c'] = 'first'
    df.loc[df['income_total'] <= 2.25e5, 'income_c'] = 'second'
    df.loc[df['income_total'] <= 1.575e5, 'income_c'] = 'third'
    df.loc[df['income_total'] <= 1.215e5, 'income_c'] = 'fourth'

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

    df.drop(columns=['begin_month', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'income_c'], inplace=True)

    # encoding
    le_gender = LabelEncoder()
    le_car = LabelEncoder()
    le_reality = LabelEncoder()
    le_income_type = LabelEncoder()
    le_edu_type = LabelEncoder()
    le_family_type = LabelEncoder()
    le_house_type = LabelEncoder()

    new_df = deepcopy(df)

    new_df.gender = le_gender.fit_transform(new_df.gender)
    new_df.car = le_car.fit_transform(new_df.car)
    new_df.reality = le_reality.fit_transform(new_df.reality)
    new_df.income_type = le_income_type.fit_transform(new_df.income_type)
    new_df.edu_type = le_edu_type.fit_transform(new_df.edu_type)
    new_df.family_type = le_family_type.fit_transform(new_df.family_type)
    new_df.house_type = le_house_type.fit_transform(new_df.house_type)

    if config.imp :
        le_occyp_type = LabelEncoder()
        new_df.occyp_type = le_occyp_type.fit_transform(new_df.occyp_type)

    # scaling
    ss_income_total = StandardScaler()
    ss_days_birth = StandardScaler()
    ss_days_employed = StandardScaler()
    ss_begin_month = StandardScaler()

    new_df.income_total = ss_income_total.fit_transform(np.array(new_df.income_total).reshape(-1, 1))
    new_df.age = ss_days_birth.fit_transform(np.array(new_df.age).reshape(-1, 1))
    new_df.skill = ss_days_employed.fit_transform(np.array(new_df.skill).reshape(-1, 1))
    new_df.month = ss_begin_month.fit_transform(np.array(new_df.month).reshape(-1, 1))

    if config.objective == 'ae':
        column_names = ['gender', 'car', 'reality', \
                         'FLAG_MOBIL', 'work_phone', 'phone', 'email']

        new_df = new_df[column_names].values

        train_df = new_df[:26457]
        test_df = new_df[26457 :]

        train_X = train_df
        train_y = train_df_target.values
        test_X = test_df

        return train_X, train_y, test_X, new_df


    elif config.objective == 'normal':
        train_df = new_df[:26457]
        test_df = new_df[26457:]

        train_X = train_df.values[:, 1:]
        train_y = train_df_target.values
        test_X = test_df.values[:, 1:]

        return train_X, train_y, test_X


def get_loaders(config):
    from sklearn.model_selection import train_test_split


    x, y, _, _ = get_data(config)
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
