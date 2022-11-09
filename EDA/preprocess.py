import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_values(value):
    return value.values.reshape(-1, 1)


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

categorical_features = ['COMPONENT_ARBITRARY', 'YEAR']

train = train.fillna(0)
test = test.fillna(0)

X = train.drop(['ID'], axis = 1)
y = train['Y_LABEL']

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for col in train_X.columns:
    if col not in categorical_features:
        scaler = StandardScaler()
        train_X[col] = scaler.fit_transform(get_values(train_X[col]))
        val_X[col] = scaler.transform(get_values(val_X[col]))
        if col in test.columns:
            test[col] = scaler.transform(get_values(test[col]))

le = LabelEncoder()
for col in categorical_features:
    train_X[col] = le.fit_transform(train_X[col])
    val_X[col] = le.transform(val_X[col])
    if col in test.columns:
        test[col] = le.transform(test[col])

