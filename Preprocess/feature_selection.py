from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def get_values(value):
    return value.values.reshape(-1, 1)


train = pd.read_csv('../data/train_after.csv')
X = train.drop(['ID', 'Y_LABEL'], axis = 1)
y = train['Y_LABEL']

print(f'Total Features: {X.shape[1]}')
categorical_features = ['COMPONENT_ARBITRARY', 'YEAR']
le = LabelEncoder()
for col in categorical_features:
    X[col] = le.fit_transform(X[col])

X = X.fillna(0)


for col in X.columns:
    if col not in categorical_features:
        scaler = MinMaxScaler()
        X[col] = scaler.fit_transform(get_values(X[col]))


rfe_selector = RFE(estimator=LogisticRegression(max_iter=10000), n_features_to_select=250, step=10, verbose=5)
rfe_selector.fit(X, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features by LR:', list(set(X.columns) - set(rfe_feature)))


chi_selector = SelectKBest(chi2, k='all')
chi_selector.fit(X, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features by chi:', list(set(X.columns) - set(chi_feature)))


embeded_lr_selector = SelectFromModel(estimator=LogisticRegression(max_iter=10000))
embeded_lr_selector.fit(X, y)
embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features by Embeded:', list(set(X.columns) - set(embeded_lr_feature)))

