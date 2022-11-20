import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import re
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


def get_values(value):
    return value.values.reshape(-1, 1)



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in train.columns}
new_n_list = list(new_names.values())
# [LightGBM] Feature appears more than one time.
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
train = train.rename(columns=new_names)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_for_LR = np.zeros((len(train), 3))

categorical_features = ['COMPONENT_ARBITRARY', 'YEAR']
# Inference(실제 진단 환경)에 사용하는 컬럼
test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR', 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']

train = train.fillna(0)
test = test.fillna(0)

x_origin = train.drop(['ID', 'Y_LABEL'], axis = 1)
y_origin = train['Y_LABEL']

test = test.drop(['ID'], axis = 1)
score_list = []
for tr_idx, val_idx in folds.split(x_origin, y_origin):
    x_d = deepcopy(x_origin)
    y_d = deepcopy(y_origin)
    test_d = deepcopy(test)
    x_train, x_val = x_d.iloc[tr_idx], x_d.iloc[val_idx]
    y_train, y_val = y_d.iloc[tr_idx], y_d.iloc[val_idx]

    # Feature Engineering
    le = LabelEncoder()
    for col in categorical_features:
        x_train[col] = le.fit_transform(x_train[col])
        x_val[col] = le.transform(x_val[col])
        if col in test_d.columns:
            test_d[col] = le.transform(test_d[col])

    params = {}
    params['objective'] = 'binary'
    params["verbose"] = 1
    params['metric'] = 'binary_logloss'
    params['device_type'] = 'gpu'
    params['boosting_type'] = 'gbdt'
    params['learning_rate'] = 0.0037695694179783445  # 0.013119로 고치면 댐
    # 예측력 상승
    params['num_iterations'] = 2500  # = num round, num_boost_round
    params['min_child_samples'] = 133
    params['n_estimators'] = 9989  # 8500
    params['num_leaves'] = 19916
    params['max_depth'] = 41  # 26?
    # overfitting 방지
    params['min_child_weight'] = 2.132006838223528  # 높을수록 / 최대 6?
    params['subsample'] = 0.9867667532255102  # 낮을수록 overfitting down / 최소 0  = bagging_fraction
    params['subsample_freq'] = 63
    params['reg_alpha'] = 0.26753664302600466  # = lambda l1
    params['reg_lambda'] = 0.10282992873487086  # = lambda l2
    params['min_gain_to_split'] = 0.5613968110180947  # = min_split_gain
    params['colsample_bytree'] = 0.9039390685690392  # 낮을 수록 overfitting down / 최소 0  = feature_fraction

    bst = lgb.LGBMClassifier(**params)
    bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='binary_logloss', early_stopping_rounds=25)

    pred = bst.predict(x_val, num_iteration=bst.best_iteration_)
    pred_proba = bst.predict_proba(x_val, num_iteration=bst.best_iteration_)

    cash = np.concatenate((pred_proba,pred.reshape(len(pred),1)), axis=1)
    y_for_LR[val_idx] = cash
    f1_macro = competition_metric(y_val, pred)
    print('The f1_macro of prediction is:', f1_macro)
    score_list.append(f1_macro)

df = pd.DataFrame(y_for_LR)
df.to_csv('LGBM_oil_train_predict.csv', index=False)
print(score_list)
