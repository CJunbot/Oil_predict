import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from lightgbm import plot_importance
import matplotlib.pyplot as plt

train = pd.read_csv('../data/train.csv')

y = train['Y_LABEL']
x = train.drop(columns=['Y_LABEL','ID', 'COMPONENT_ARBITRARY'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)


params = {}
params["verbose"] = 1
params['metric'] = 'auc'
params['device_type'] = 'gpu'
params['boosting_type'] = 'gbdt'


bst = LGBMClassifier(**params)
bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='auc', early_stopping_rounds=25)
fig, ax = plt.subplots(figsize=(12,6))
plot_importance(bst, max_num_features=60, ax=ax)
plt.show()