import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
pd.set_option('display.max_columns', 500)

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

plt.figure(figsize=(70,70))
sns.heatmap(train.corr(), annot=True, annot_kws={"size": 7})
plt.show()

print(f'shape of data: train:{train.shape} test: {test.shape}\n--------------------------------------')

train['Y_LABEL'].value_counts().plot.bar()
plt.title('Y_LABEL')

print('Null data:')
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending=False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_train_data)
print('\n------------------------------------------')

print('Correlation with Y-LABEL:')
corr = train.corr()['Y_LABEL']
print(abs(corr).sort_values(ascending=False))
print('\n--------------------------------------')

fig, axes = plt.subplots(2, 1, figsize=(20,20), constrained_layout=True)
sns.countplot(x='COMPONENT_ARBITRARY', hue='Y_LABEL',data=train, ax=axes[0])
sns.countplot(x='YEAR', hue='Y_LABEL',data=train, ax=axes[1])
plt.show()

train = train.fillna(0)
def plot_category_percent_of_target(col):
    fig, ax = plt.subplots(1, 1, figsize=(30, 20), constrained_layout=True)
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
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
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
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
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


numeric_features = ['ANONYMOUS_1','YEAR','SAMPLE_TRANSFER_DAY','ANONYMOUS_2','AG','AL','B','BA','BE','CA','CD','CO','CR','CU','FH2O','FNOX','FOPTIMETHGLY','FOXID','FSO4','FTBN','FE','FUEL','H2O','K','LI','MG','MN','MO','NA','NI','P','PB','PQINDEX','S','SB','SI','SN','SOOTPERCENTAGE','TI','U100','U75','U50','U25','U20','U14','U6','U4','V','V100','V40','ZN'
]
i = 0
for col in numeric_features:
        plot_kde_hist_for_numeric(col)
        plot_category_percent_of_target_for_numeric(col)
        i += 1

plot_kde_hist_for_numeric('SAMPLE_TRANSFER_DAY')

print(f'\n{i}개 columns plot done\n------------------------------------')

