import optuna
from optuna.samplers import TPESampler
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import re
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


sampler = TPESampler(seed=10)

def objective(trial):
    train = pd.read_csv('train.csv')

    new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in train.columns}
    new_n_list = list(new_names.values())
    # [LightGBM] Feature appears more than one time.
    new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in
                 enumerate(new_names.items())}
    train = train.rename(columns=new_names)

    categorical_features = ['COMPONENT_ARBITRARY']
    # Inference(실제 진단 환경)에 사용하는 컬럼

    train = train.fillna(0)
    two_Y = train['Y_LABEL']
    all_X = train.drop(['ID', 'Y_LABEL'], axis=1)

    x_train, x_val, y_train, y_val = train_test_split(all_X, two_Y, test_size=0.2, random_state=42, stratify=two_Y)

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
    params['random_state'] = 42
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.001, 0.005)
    # 예측력 상승
    params['num_iterations'] = 5000
    params['min_child_samples'] = trial.suggest_int('min_child_samples', 1, 100)
    params['n_estimators'] = trial.suggest_int('n_estimators',10, 3000)
    params['num_leaves'] = trial.suggest_int('num_leaves', 100, 2505)
    params['max_depth'] = trial.suggest_int('max_depth', 20, 250)
    # overfitting 방지
    params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.6, 0.99)
    params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.7, 0.99)

    bst = lgb.train(params, train_data, 5000, [val_data], verbose_eval=5, early_stopping_rounds=25)

    # pred_proba = bst.predict_proba(x_val, num_iteration=bst.best_iteration_)
    # cash = np.concatenate((pred_proba,pred.reshape(len(pred),1)), axis=1)
    best_macro = 0
    for i in [0.05, 0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        pred = bst.predict(x_val)
        preds = np.where(np.array(pred) > i, 1, 0)
        f1_macro = competition_metric(y_val, preds)
        if f1_macro > best_macro:
            best_macro = f1_macro
    return best_macro


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=120)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

