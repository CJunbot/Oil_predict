import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import re
import numpy as np
from sklearn.multioutput import MultiOutputRegressor


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


train = pd.read_csv('../data/train_after.csv')
test = pd.read_csv('../data/test.csv')
knowledge = pd.read_csv('teacher_4cnn.csv')

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
all_X = all_X[test_stage_features]

two_Y = np.log1p(knowledge['0'])

test = test.drop(['ID'], axis=1)

train_X, val_X, train_y, val_y = train_test_split(all_X, two_Y, test_size=0.2, random_state=42)


def get_values(value):
    return value.values.reshape(-1, 1)


le = LabelEncoder()
for col in categorical_features:
    train_X[col] = le.fit_transform(train_X[col])
    val_X[col] = le.transform(val_X[col])
    if col in test.columns:
        test[col] = le.transform(test[col])

params = {}
params['objective'] = 'regression'
params["verbose"] = 1
params['metric'] = 'l1'
params['device_type'] = 'gpu'
params['boosting_type'] = 'gbdt'
params['learning_rate'] = 0.0048695694179783445  # 0.013119로 고치면 댐
# 예측력 상승
params['num_iterations'] = 2500  # = num round, num_boost_round
params['min_child_samples'] = 121
params['n_estimators'] = 13553  # 8500
params['num_leaves'] = 677
params['max_depth'] = 10  # 26?
# overfitting 방지
params['min_child_weight'] = 0.6362716374372734  # 높을수록 / 최대 6?
params['min_child_samples'] = 16  # 100 500 ?
params['subsample'] = 0.8123575694329729  # 낮을수록 overfitting down / 최소 0  = bagging_fraction
params['subsample_freq'] = 84
params['reg_alpha'] = 1.2469938020520392  # = lambda l1
params['reg_lambda'] = 0.47632935403032983  # = lambda l2
params['min_gain_to_split'] = 0.07328373339731083  # = min_split_gain
params['colsample_bytree'] = 0.8686525213979316  # 낮을 수록 overfitting down / 최소 0  = feature_fraction

bst = lgb.LGBMRegressor(**params)
bst.fit(train_X, train_y, eval_set=[(val_X, val_y)], eval_metric='l1', early_stopping_rounds=25)
pred = np.expm1(bst.predict(test))

df = pd.DataFrame(pred)
df.to_csv('pred0.csv', index=False)
