import random

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold


def classify(train, test, k, save_fn, config):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for train_idx, valid_idx in skf.split(train, train['credit']) :
        folds.append((train_idx, valid_idx))

    random.seed(42)
    models = {}
    scores = []
    for fold in range(k) :
        print(f'===================================={fold + 1}============================================')
        train_idx, valid_idx = folds[fold]
        X_train, X_valid, y_train, y_valid = train.drop(['credit'], axis=1).iloc[train_idx].values, \
                                             train.drop(['credit'], axis=1).iloc[valid_idx].values, \
                                             train['credit'][train_idx].values, train['credit'][valid_idx].values

        if config.model_name == 'lgb':
            lgb = LGBMClassifier(
                application='multiclass',
                boosting='gbdt',

                max_depth=17,
                num_leaves=90, #90
                min_data_in_leaf=19,  #19

                num_iterations=3000,
                early_stopping_rounds=30,
                learning_rate=0.01,

                min_sum_hessian_in_leaf=3,
                bagging_fraction=0.9,
                bagging_freq=1,
                feature_fraction=0.6,
                lambda_l2=0.7,
            )
            lgb.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)],
                    verbose=100)
            models[fold] = lgb
            scores.append(lgb.best_score_['valid_1']['multi_logloss'])

        elif config.model_name == 'xgb':
            xgb = XGBClassifier(
                objective='multi:softmax',
                grow_policy='lossguide',

                num_class=3,
                learning_rate=0.01,  # 0.005
                n_estimators=3000,
                #         patience=30,

                max_depth=15,
                min_child_weight=3,
                gamma=0.3,
                subsample=0.9,
                colsample_bytree=0.6,
                reg_alpha=1,
            )
            xgb.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)],
                    eval_metric='mlogloss',
                    early_stopping_rounds=30,
                    verbose=100)
            models[fold] = xgb
            scores.append(xgb.best_score)
        print(f'================================================================================\n\n')
    print('-' * 50, 'Result', '-' * 50, '\n')
    print('Mean : {}'.format(np.mean(scores, axis=0)))
    print('Variance : {}'.format(np.var(scores, axis=0)))
    print('_' * 106)

    submit = pd.read_csv('data/sample_submission.csv')
    submit.iloc[:, 1 :] = 0
    for fold in range(k) :
        submit.iloc[:, 1 :] += models[fold].predict_proba(test) / k

    submit.to_csv('submission/lgbm/{}'.format(save_fn), index=False)