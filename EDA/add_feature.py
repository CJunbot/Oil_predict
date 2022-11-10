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

train = train.fillna(0)
test = test.fillna(0)

train = train.drop(['ID', 'Y_LABEL'], axis=1)

train.loc[( )]
