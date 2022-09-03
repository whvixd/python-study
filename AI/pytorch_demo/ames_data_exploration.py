import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler


# 读取数据集
df_train = pd.read_csv('../../test/sources/data/ames_house/train.csv')
test_data = pd.read_csv('../../test/sources/data/ames_house/test.csv')


def print_columns():
    #check the decoration
    print(df_train.columns)

def explore_SalePrice():
    # descriptive statistics summary
    print(df_train['SalePrice'].describe())
    # 直方图
    sns.distplot(df_train['SalePrice'])

    # 偏斜系数 and 风度系数
    print("Skewness: %f" % df_train['SalePrice'].skew())
    print("Kurtosis: %f" % df_train['SalePrice'].kurt())

def scatter_plot():
    # GrLivArea与SalePrice 的散点图
    var = 'GrLivArea'
    var = 'TotalBsmtSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


def box_plot():
    # 箱线图
    var = 'OverallQual'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y='SalePrice', data=data)
    fig.axis(ymin=0, ymax=800000)

    var = 'YearBuilt'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)

def heatmap1():
    # 热图：热图是一个以颜色变化来显示数据的矩阵。
    # 此时颜色代表的就是相关系数的大小。所以可以看到自己和自己的相关系数是1，
    # 也就是最深的蓝色。约接近白色说明相关性越弱，偏蓝（正相关）或者偏红（负相关）则代表相关性强。根据右侧展示的颜色系数
    # correlation matrix
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)


def heatmap2():
    # 热图：热图是一个以颜色变化来显示数据的矩阵。
    # 此时颜色代表的就是相关系数的大小。所以可以看到自己和自己的相关系数是1，
    # 也就是最深的蓝色。约接近白色说明相关性越弱，偏蓝（正相关）或者偏红（负相关）则代表相关性强。根据右侧展示的颜色系数
    # correlation matrix
    corrmat = df_train.corr()

    # saleprice correlation matrix
    k = 10  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()

def mul_scatter_plot():
    # 组合散点图
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df_train[cols], size=2.5)
    plt.show()

def missing_data():
    # 丢失数据分析
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))

def missing_data1():
    # 丢失数据分析
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))

def dealing_missing_data():
    # 删除丢失的数据
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    df_train_drop=df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
    df_train_drop = df_train_drop.drop(df_train.loc[df_train['Electrical'].isnull()].index)
    print(df_train_drop.isnull().sum().max())

def standardizing_data():
    # 标准化数据
    saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])
    low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
    high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)

def histogram_and_plot():
    # histogram and normal probability plot
    sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
    fig = plt.figure()
    # 默认检测是正态分布
    res = stats.probplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)

if __name__ == '__main__':
    # print_columns()
    # explore_SalePrice()
    # scatter_plot()
    # box_plot()
    # heatmap2()
    # mul_scatter_plot()
    # dealing_missing_data()
    histogram_and_plot()