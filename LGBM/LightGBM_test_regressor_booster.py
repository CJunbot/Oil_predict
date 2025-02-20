import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import re
import numpy as np
from scipy.stats import skew


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
knowledge = pd.read_csv('LGBM_oil_train_predict_nofold.csv')

new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in train.columns}
new_n_list = list(new_names.values())
# [LightGBM] Feature appears more than one time.
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
train = train.rename(columns=new_names)

categorical_features = ['COMPONENT_ARBITRARY']
# Inference(실제 진단 환경)에 사용하는 컬럼
test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR' , 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']

train = train.fillna(0)
test = test.fillna(0)

all_X = train.drop(['ID', 'Y_LABEL'], axis = 1)
x_origin = all_X[test_stage_features]
y_origin = train['Y_LABEL']
test = test.drop(['ID'], axis=1)

knowledge['0'] = np.log1p(knowledge['0'])
x_train, x_val, y_train, y_val = train_test_split(x_origin, knowledge, test_size=0.2, random_state=42)

y_train = y_train['0']
label = y_val['1']
y_val = y_val['0']


def get_values(value):
    return value.values.reshape(-1, 1)


le = LabelEncoder()
for col in categorical_features:
    x_train[col] = le.fit_transform(x_train[col])
    x_val[col] = le.transform(x_val[col])
    if col in test.columns:
        test[col] = le.transform(test[col])

params = {}
params['objective'] = 'regression'
params["verbose"] = 1
params['metric'] = 'l1'
params['device_type'] = 'gpu'
params['boosting_type'] = 'gbdt'
params['random_state'] = 42
params['learning_rate'] = 0.004735180883077572  # 0.013119로 고치면 댐
# 예측력 상승
params['num_iterations'] = 2500  # = num round, num_boost_round
params['n_estimators'] = 2666  # 8500
params['num_leaves'] = 3358
params['max_depth'] = 6  # 26?
# overfitting 방지
params['min_child_samples'] = 365  # 100 500 ?
params['subsample'] = 0.8654652273719451  # 낮을수록 overfitting down / 최소 0  = bagging_fraction



params['colsample_bytree'] = 0.7734619838566695  # 낮을 수록 overfitting down / 최소 0  = feature_fraction

bst = lgb.LGBMRegressor(**params)
bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='l1', early_stopping_rounds=25)
pred = bst.predict(test, num_iteration=bst.best_iteration_)

for i in [0.05, 0.1, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    preds = np.expm1(bst.predict(x_val, num_iteration=bst.best_iteration_))
    preds = np.where(np.array(preds) > i, 1, 0)
    f1_macro = competition_metric(label, preds)
    print('The f1_macro of prediction is:', {i}, f1_macro)

df = pd.DataFrame(np.expm1(pred))
df.to_csv('pred1.csv', index=False)
