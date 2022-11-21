import optuna
from optuna.samplers import TPESampler
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import re
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


sampler = TPESampler(seed=10)
param = []


def objective(trial):
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    knowledge = pd.read_csv('LGBM_oil_train_predict.csv')

    new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in train.columns}
    new_n_list = list(new_names.values())
    # [LightGBM] Feature appears more than one time.
    new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in
                 enumerate(new_names.items())}
    train = train.rename(columns=new_names)

    categorical_features = ['COMPONENT_ARBITRARY']
    # Inference(실제 진단 환경)에 사용하는 컬럼
    test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR', 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE',
                           'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']

    train = train.fillna(0)
    test = test.fillna(0)

    all_X = train.drop(['ID', 'Y_LABEL'], axis=1)
    all_X = all_X[test_stage_features]

    test = test.drop(['ID'], axis=1)

    train_X, val_X, train_y, val_y = train_test_split(all_X, knowledge, test_size=0.2, random_state=42)

    train_y = train_y['0']
    label = val_y['1']
    val_y = val_y['0']


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
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.001, 0.005)
    # 예측력 상승
    params['num_iterations'] = 50000
    params['n_estimators'] = trial.suggest_int('n_estimators', 20, 10000)
    params['num_leaves'] = trial.suggest_int('num_leaves', 32, 10000)
    params['max_depth'] = trial.suggest_int('max_depth', 5, 20)
    # overfitting 방지
    params['min_child_weight'] = trial.suggest_float('min_child_weight', 0.01, 4)
    params['min_child_samples'] = trial.suggest_int('min_child_samples', 1, 1000)
    params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.6, 0.99)
    params['subsample_freq'] = trial.suggest_int('subsample_freq', 60, 99)
    params['lambda_l1'] = trial.suggest_float('lambda_l1', 0.01, 2)
    params['lambda_l2'] = trial.suggest_float('lambda_l2', 0.01, 2)
    params['min_gain_to_split'] = trial.suggest_float('min_gain_to_split', 0.001, 0.8)
    params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.7, 0.99)

    bst = lgb.LGBMRegressor(**params)
    bst.fit(train_X, train_y, eval_set=[(val_X, val_y)], eval_metric='l1', early_stopping_rounds=25)
    best_macro = 0
    for i in [0.05, 0.1, 0.15, 0.17, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        preds = bst.predict(val_X, num_iteration=bst.best_iteration_)
        preds = np.where(np.array(preds) > i, 1, 0)
        f1_macro = competition_metric(label, preds)
        if f1_macro > best_macro:
            best_macro = f1_macro
    return f1_macro


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))