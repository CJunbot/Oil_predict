import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train = pd.read_csv('../data/train_after.csv')
test = pd.read_csv('../data/test.csv')

# continuous feature
test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR' , 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']

fig, axes = plt.subplots(4, 2, figsize=(25,35), constrained_layout=True)
sns.countplot(x='COMPONENT_ARBITRARY', data=train, ax=axes[0][0])
sns.countplot(x='COMPONENT_ARBITRARY', data=test, ax=axes[0][1])
sns.histplot(x='ANONYMOUS_1', kde=True, data=train, ax=axes[1][0], log_scale=True)
sns.histplot(x='ANONYMOUS_1', kde=True, data=test, ax=axes[1][1], log_scale=True)
sns.countplot(x='YEAR', data=train, ax=axes[2][0])
sns.countplot(x='YEAR', data=test, ax=axes[2][1])
sns.histplot(x='ANONYMOUS_2', data=train, kde=True,  ax=axes[3][0])
sns.histplot(x='ANONYMOUS_2', data=test, kde=True, ax=axes[3][1])
plt.show()

fig, axes = plt.subplots(5, 2, figsize=(25,35), constrained_layout=True)
sns.countplot(x='AG', data=train, ax=axes[0][0])
sns.countplot(x='AG', data=test, ax=axes[0][1])
sns.countplot(x='CO', data=train, ax=axes[1][0])
sns.countplot(x='CO', data=test, ax=axes[1][1])
sns.countplot(x='CR', data=train, ax=axes[2][0])
sns.countplot(x='CR', data=test, ax=axes[2][1])
sns.countplot(x='CU', data=train, ax=axes[3][0])
sns.countplot(x='CU', data=test, ax=axes[3][1])
sns.countplot(x='FE', data=train, ax=axes[4][0])
sns.countplot(x='FE', data=test, ax=axes[4][1])
plt.show()

fig, axes = plt.subplots(4, 2, figsize=(25,35), constrained_layout=True)
sns.countplot(x='H2O', data=train, ax=axes[0][0])
sns.countplot(x='H2O', data=test, ax=axes[0][1])
sns.countplot(x='MN', data=train, ax=axes[1][0])
sns.countplot(x='MN', data=test, ax=axes[1][1])
sns.countplot(x='MO', data=train, ax=axes[2][0])
sns.countplot(x='MO', data=test, ax=axes[2][1])
sns.countplot(x='NI', data=train, ax=axes[3][0])
sns.countplot(x='NI', data=test, ax=axes[3][1])
plt.show()

fig, axes = plt.subplots(5, 2, figsize=(25,35), constrained_layout=True)
sns.countplot(x='PQINDEX', data=train, ax=axes[0][0])
sns.countplot(x='PQINDEX', data=test, ax=axes[0][1])
sns.countplot(x='TI', data=train, ax=axes[1][0])
sns.countplot(x='TI', data=test, ax=axes[1][1])
sns.countplot(x= 'V', data=train, ax=axes[2][0])
sns.countplot(x= 'V', data=test, ax=axes[2][1])
sns.histplot(x='V40', data=train, kde=True, ax=axes[3][0])
sns.histplot(x='V40', data=test, kde=True, ax=axes[3][1])
sns.histplot(x='ZN', data=train, kde=True, ax=axes[4][0])
sns.histplot(x='ZN', data=test, kde=True, ax=axes[4][1])
plt.show()
