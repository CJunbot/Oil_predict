import pandas as pd
import lightgbm as lgb
from lightgbm import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import re
import numpy as np

def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


train = pd.read_csv('../data/train_after.csv')
test = pd.read_csv('../data/test.csv')

new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in train.columns}
new_n_list = list(new_names.values())
# [LightGBM] Feature appears more than one time.
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
train = train.rename(columns=new_names)

categorical_features = ['COMPONENT_ARBITRARY', 'YEAR']
# Inference(실제 진단 환경)에 사용하는 컬럼
test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR' , 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']

train = train.fillna(0)
test = test.fillna(0)

all_X = train.drop(['ID', 'Y_LABEL'], axis = 1)
all_y = train['Y_LABEL']

test = test.drop(['ID'], axis = 1)

train_X, val_X, train_y, val_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42, stratify=all_y)


def get_values(value):
    return value.values.reshape(-1, 1)


le = LabelEncoder()
for col in categorical_features:
    train_X[col] = le.fit_transform(train_X[col])
    val_X[col] = le.transform(val_X[col])
    if col in test.columns:
        test[col] = le.transform(test[col])

params = {}
params['objective'] = 'binary'
params["verbose"] = 1
params['metric'] = 'binary_logloss'
params['device_type'] = 'gpu'
params['boosting_type'] = 'gbdt'
params['learning_rate'] = 0.005257799871701795  # 0.013119로 고치면 댐
# 예측력 상승
params['num_iterations'] = 2500  # = num round, num_boost_round
params['min_child_samples'] = 548
params['n_estimators'] = 16087  # 8500
params['num_leaves'] = 7717
params['max_depth'] = 187  # 26?
# overfitting 방지
params['min_child_weight'] = 1.6974077894395867  # 높을수록 / 최대 6?
params['min_child_samples'] = 16  # 100 500 ?
params['subsample'] = 0.9891861796802488  # 낮을수록 overfitting down / 최소 0  = bagging_fraction
params['subsample_freq'] = 60
params['reg_alpha'] = 1.9632489581094166  # = lambda l1
params['reg_lambda'] = 0.9522680494417242  # = lambda l2
params['min_gain_to_split'] = 0.015443901171054597  # = min_split_gain
params['colsample_bytree'] = 0.9393774324594517  # 낮을 수록 overfitting down / 최소 0  = feature_fraction

bst = lgb.LGBMClassifier(**params)
bst.fit(train_X, train_y, eval_set=[(val_X, val_y)], eval_metric='binary_logloss', early_stopping_rounds=25)
pred = bst.predict_proba(val_X, num_iteration=bst.best_iteration_)
print(pred)
pred = bst.predict(val_X, num_iteration=bst.best_iteration_)
f1_macro = competition_metric(val_y, pred)
print('The f1_macro of prediction is:', f1_macro)

#fig, ax = plt.subplots(figsize=(60,60))
#plot_importance(bst, max_num_features=204, ax=ax)
#plt.show()
