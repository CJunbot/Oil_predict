import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

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
params['learning_rate'] = 0.013119110575691373  # 0.013119로 고치면 댐
# 예측력 상승
params['num_iterations'] = 2500  # = num round, num_boost_round
params['min_child_samples'] = 118
params['n_estimators'] = 15918  # 8500
params['num_leaves'] = 7868
params['max_depth'] = 35  # 26?
# overfitting 방지
params['min_child_weight'] = 0.7628373492320147  # 높을수록 / 최대 6?
params['min_child_samples'] = 41  # 100 500 ?
#params['subsample'] = 0.7611163934517731  # 낮을수록 overfitting down / 최소 0  = bagging_fraction
params['subsample_freq'] = 76
params['reg_alpha'] = 0.46641059279049957  # = lambda l1
params['reg_lambda'] = 0.30503746605875  # = lambda l2
params['min_gain_to_split'] = 0.05443147365335205  # = min_split_gain
params['colsample_bytree'] = 0.9009386979948221  # 낮을 수록 overfitting down / 최소 0  = feature_fraction

bst = lgb.LGBMClassifier(**params)
bst.fit(train_X, train_y, eval_set=[(val_X, val_y)], eval_metric='binary_logloss', early_stopping_rounds=25)
pred = bst.predict(val_X, num_iteration=bst.best_iteration_)
MAE = mean_absolute_error(val_y, pred)
print('The MAE of prediction is:', MAE)

pred = bst.predict(test, num_iteration=bst.best_iteration_)

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = pred
sample_submission.to_csv("../data/submit_lgbm.csv", index=False)

bst.booster_.save_model('model2.txt')