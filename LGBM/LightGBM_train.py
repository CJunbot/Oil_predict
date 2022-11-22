import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from sklearn.model_selection import train_test_split
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

categorical_features = ['COMPONENT_ARBITRARY']  # YEAR
# Inference(실제 진단 환경)에 사용하는 컬럼
test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR', 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']

train = train.fillna(0)
test = test.fillna(0)

x_origin = train.drop(['ID', 'Y_LABEL'], axis = 1)
y_origin = train['Y_LABEL']

test = test.drop(['ID'], axis = 1)
score_list = []
x_train, x_val, y_train, y_val = train_test_split(x_origin, y_origin, test_size=0.2, random_state=42, stratify=y_origin)


# Feature Engineering
le = LabelEncoder()
for col in categorical_features:
    x_train[col] = le.fit_transform(x_train[col])
    x_val[col] = le.transform(x_val[col])

train_data = lgb.Dataset(x_train, label=y_train, categorical_feature=['COMPONENT_ARBITRARY', 'YEAR'])
val_data = lgb.Dataset(x_val, label=y_val, categorical_feature=['COMPONENT_ARBITRARY', 'YEAR'])

params = {}
params['objective'] = 'binary'
params["verbose"] = 1
params['metric'] = 'binary_logloss'
params['device_type'] = 'gpu'
params['boosting_type'] = 'gbdt'
params['learning_rate'] = 0.0037695694179783445  # 0.013119로 고치면 댐
# 예측력 상승
params['num_iterations'] = 5500  # = num round, num_boost_round
params['min_child_samples'] = 30
params['num_leaves'] = 128
params['random_state'] = 42
# overfitting 방지
params['subsample'] = 0.8  # 낮을수록 overfitting down / 최소 0  = bagging_fraction

params['colsample_bytree'] = 0.7  # 낮을 수록 overfitting down / 최소 0  = feature_fraction

#bst = lgb.LGBMClassifier(**params)
#bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='binary_logloss', early_stopping_rounds=5)
bst = lgb.train(params, train_data, 5000, [val_data], verbose_eval=5, early_stopping_rounds=25)

#pred_proba = bst.predict_proba(x_val, num_iteration=bst.best_iteration_)
#cash = np.concatenate((pred_proba,pred.reshape(len(pred),1)), axis=1)
best_macro = 0
for i in [0.05, 0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    pred = bst.predict(x_val)
    preds = np.where(np.array(pred) > i, 1, 0)
    f1_macro = competition_metric(y_val, preds)
    if f1_macro > best_macro:
        best_macro = f1_macro

print('The f1_macro of prediction is:', best_macro)

le = LabelEncoder()
for col in categorical_features:
    x_origin[col] = le.fit_transform(x_origin[col])

pred = bst.predict(x_origin)
y_origin = np.array(y_origin)
df = np.concatenate((pred.reshape(len(pred), 1), y_origin.reshape(len(y_origin),1)), axis=1)
df = pd.DataFrame(df)
df.to_csv('LGBM_oil_train_predict_nofold.csv', index=False)
