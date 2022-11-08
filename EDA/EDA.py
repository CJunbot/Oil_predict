import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics

plt.style.use('seaborn')
sns.set(font_scale=2)
pd.set_option('display.max_columns', 500)

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
print('shape of data\n--------------------------------------')
print(train.shape, test.shape)

train['Y_LABEL'].value_counts().plot.bar()
plt.title('Y_LABEL')

print('null data\n--------------------------------------')
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_train_data)

fig, axes = plt.subplots(2, 1, figsize=(20,20))
sns.countplot(x='COMPONENT_ARBITRARY', hue='Y_LABEL',data=train, ax=axes[0])
sns.countplot(x='YEAR', hue='Y_LABEL',data=train, ax=axes[1])
plt.show()

train = train.fillna(0)
def plot_category_percent_of_target(col):
    fig, ax = plt.subplots(1, 1, figsize=(30, 20))
    cat_percent = train[[col, 'Y_LABEL']].groupby(col, as_index=False).mean()
    cat_size = train[col].value_counts().reset_index(drop=False)
    cat_size.columns = [col, 'count']
    cat_percent = cat_percent.merge(cat_size, on=col, how='left')
    cat_percent['Y_LABEL'] = cat_percent['Y_LABEL'].fillna(0)
    cat_percent = cat_percent.sort_values(by='count', ascending=False)[:20]
    sns.barplot(ax=ax, x='Y_LABEL', y=col, data=cat_percent, order=cat_percent[col])

    for i, p in enumerate(ax.patches):
        ax.annotate('{}'.format(cat_percent['count'].values[i]), (p.get_width(), p.get_y()+0.5), fontsize=20)

    plt.xlabel('% of Y_LABEL')
    plt.ylabel(col)
    plt.show()

plot_category_percent_of_target('COMPONENT_ARBITRARY')


def plot_kde_hist_for_numeric(col):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.kdeplot(train.loc[train['Y_LABEL'] == 0, col], ax=ax[0], label='Y_LABEL(0)')
    sns.kdeplot(train.loc[train['Y_LABEL'] == 1, col], ax=ax[0], label='Y_LABEL(1)')

    train.loc[train['Y_LABEL'] == 0, col].hist(ax=ax[1], bins=100)
    train.loc[train['Y_LABEL'] == 1, col].hist(ax=ax[1], bins=100)

    plt.suptitle(col, fontsize=30)
    try:
        ax[0].set_yscale('log')
        ax[0].set_title('KDE plot')
    except:
        ax[0].set_title('KDE plot')

    ax[1].set_title('Histogram')
    ax[1].legend(['Y_LABEL-0', 'Y_LABEL-1'])
    try:
        ax[1].set_yscale('log')
    except:
        plt.show()
    plt.show()

def plot_category_percent_of_target_for_numeric(col):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    cat_percent = train[[col, 'Y_LABEL']].groupby(col, as_index=False).mean()
    cat_size = train[col].value_counts().reset_index(drop=False)
    cat_size.columns = [col, 'count']
    cat_percent = cat_percent.merge(cat_size, on=col, how='left')
    cat_percent['Y_LABEL'] = cat_percent['Y_LABEL'].fillna(0)
    cat_percent = cat_percent.sort_values(by='count', ascending=False)[:20]
    cat_percent[col] = cat_percent[col].astype('category')
    sns.barplot(ax=ax[0], x='Y_LABEL', y=col, data=cat_percent, order=cat_percent[col])

    for i, p in enumerate(ax[0].patches):
        ax[0].annotate('{}'.format(cat_percent['count'].values[i]), (p.get_width(), p.get_y() + 0.5), fontsize=20)

    ax[0].set_title('Barplot sorted by count', fontsize=20)

    sns.barplot(ax=ax[1], x='Y_LABEL', y=col, data=cat_percent)
    for i, p in enumerate(ax[0].patches):
        ax[1].annotate('{}'.format(cat_percent['count'].sort_index().values[i]), (0, p.get_y() + 0.6), fontsize=20)
    ax[1].set_title('Barplot sorted by index', fontsize=20)

    plt.xlabel('% of Y-LABEL')
    plt.ylabel(col)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.show()


numeric_features = list(set(train.columns) - set(['YEAR', 'COMPONENT_ARBITRARY', 'ID', 'Y_LABEL']))
i = 0
for col in numeric_features:
    if train[col].sum() != 0:
        try:
            plot_kde_hist_for_numeric(col)
            plot_category_percent_of_target_for_numeric(col)
            i += 1
        except:
            sns.countplot(x=col, hue='Y_LABEL', data=train)
            plot_category_percent_of_target(col)
            i += 1
print(f'{i}개 columns plot done')
print('Correlation with Y-LABEL\n--------------------------------------')
corr = train.corr()['Y_LABEL']
print(abs(corr).sort_values(ascending=False))
