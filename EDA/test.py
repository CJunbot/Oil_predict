import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def get_values(value):
    return value.values.reshape(-1, 1)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

categorical_features = ['COMPONENT_ARBITRARY', 'YEAR']

train = train.fillna(0)
test = test.fillna(0)

X = train.drop(['ID'], axis=1)
y = train['Y_LABEL']


for col in X.columns:
    if col not in categorical_features:
        if col != 'Y_LABEL':
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(get_values(X[col]))
            if col in test.columns:
                test[col] = scaler.transform(get_values(test[col]))

le = LabelEncoder()
for col in categorical_features:
    X[col] = le.fit_transform(get_values(X[col]))
    if col in test.columns:
        test[col] = le.transform(get_values(test[col]))

print(X.head())
X.to_csv('train_after.csv', index=False)
