# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns#画图使用的

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

'''
总共有(22730, 18)训练集
总共有(15154, 17)测试集
'''
pd.pandas.set_option('display.max_columns', None)#不设置最大行的使用

# load train dataset /
path = 'E:\ZGW\PycharmProjects1\pythonProject1\Data_an\compact_solution\test.csv'
dataset = pd.read_csv("E:/ZGW/PycharmProjects1/pythonProject1/Data_an/compact_solution/train.csv")
print(dataset.shape)

val_dataset = pd.read_csv("E:/ZGW/PycharmProjects1/pythonProject1/Data_an/compact_solution/test.csv")
print(val_dataset.shape)#验证数据集的shape

#print the top of 5
# print(dataset.head(5))
# print(val_dataset.head(5))

#check for missing values
# print(dataset.columns)
# print(dataset['numberOfRooms'].isnull())
# print(dataset['numberOfRooms'].isna())
feature_with_na = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1]
for feature in feature_with_na:
    print(feature,np.round(dataset[feature].isnull().mean(),4),'% missing values')
#如果没有输出的话，那么就说明没有缺失值。

#numeric features和非数字型变量
print(type('0'))
numerical_feature = [feature for feature in dataset.columns if dataset[feature].dtypes != '0']
# print(numerical_feature)#也就是说全部的特征变量都是数值型变量，在进行划分就是离散型变量和非离散型变量。
# print(len(numerical_feature))
# print(dataset[numerical_feature].head())
# print(dataset.dtypes)
# print(dataset.describe().T)

#year variable
Yr_features = [feature for feature in numerical_feature if 'made' in feature]
print(Yr_features)
for feature in Yr_features:
    print(feature,dataset[feature].unique())
print(len(dataset[feature].unique()))
#let is plot the median house price and how it is varying
# data = dataset.groupby('made')['price'].median().plot()
# plt.xlabel('year made')
# plt.ylabel('house price')
# plt.title('year vs price')
# plt.show()
#let is see how many records we have with made = 10000
data = dataset[dataset['made'] == 10000] #总共有5条数据，需要进行删除。
dataset_without_10000_made = dataset[dataset['made'] != 10000]
# data = dataset_without_10000_made.groupby('made')['price'].median().plot()#取的是一个中位数
# plt.xlabel('year made')
# plt.ylabel('house price')
# plt.title('year vs price')
# plt.show()

for feature in Yr_features:
    print(feature,dataset[feature].unique())
print(Yr_features+['id'])


#discrete features 离散特征
discrete_features = []
#类别可数就是离散，不可数是连续
'''
连续变量，指在一定区间内可以任意取值，相邻的两个数值可作无限分割(即可取无限个值)。
比如题主所说的身高，身高可以是183，也可以是183.1，也可以是183.111……1。
离散变量，是指其数值只能用自然数、整数、计数单位等描述的数据。
例如，职工个数(总不能是1.2个吧)，成绩A+等。

作者：小浩浩
链接：https://www.zhihu.com/question/306601010/answer/614207635
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''
for feature in numerical_feature:
    if len(dataset[feature].unique()) < 25 and feature not in Yr_features+['id']:
        discrete_features.append(feature)
print(discrete_features)

# print(dataset[discrete_features].head())

#let is try to plot the median price vs discrete features and see how they are behaving
# for feature in discrete_features:
#     data = dataset.copy()
#     sns.boxplot(x=feature,y='price',data=data)#箱图
#     plt.show()
#中位数波动相对不大
#                hasYard  hasPool  cityPartRange  numPrevOwners  isNewBuilt hasStormProtector  hasStorageRoom  hasGuestRoom
# for feature in ['hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector', 'hasStorageRoom']:
#     print('median price for ',dataset.groupby(feature)['price'].median())
#     dataset.groupby(feature)['price'].median().plot.bar()#条形图
#     plt.xlabel(feature)
#     plt.ylabel('price')
#     plt.show()

#可以看出来中位数波动还是蛮大的。
# for feature in ['cityPartRange', 'numPrevOwners', 'hasGuestRoom']:
#     print('median price for',dataset.groupby(feature)['price'].median())
#     dataset.groupby(feature)['price'].median().plot()
#     plt.xlabel(feature)
#     plt.ylabel('price')
#     plt.show()

#continuous features

continuous_features = [feature for feature in numerical_feature if feature not in discrete_features and feature not in Yr_features + ['id']]
print(continuous_features)
for feature in continuous_features:
    print(feature,':',len(dataset[feature].unique()))
for feature in continuous_features:
    data = dataset.copy()
    #kde=True:表示要绘制核密度估计图(默认情况为True),若为False,则绘制
    # sns.histplot(x=feature,data=data,bins=25)#画出来直方图的
    # plt.xlabel(feature)
    # plt.ylabel('Count')
    # plt.title(feature)
    # plt.show()

import scipy.stats as stats


def diagnostic_plots(df, variable):
    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    # bins: 字符型、整型、向量都可以，可以是引用规则的名称、箱子的数量或箱子的分段或者分箱规则名称，规则名称见下方示例
    # kde: 是否生成核密度曲线
    sns.histplot(df[variable], bins=30, kde=True)  # 直方图
    plt.title('Histogram')

    # Q-Q plots
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)  # 计算概率图的分位数，并可选择显示该图。

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])  # 画出来箱图
    plt.title('Boxplot')

    plt.show()

# let's find the outliers in squaremeters
# diagnostic_plots(dataset, 'cityCode')
# diagnostic_plots(dataset, 'squareMeters')
# diagnostic_plots(dataset, 'floors')
# diagnostic_plots(dataset, 'numberOfRooms')
# diagnostic_plots(dataset, 'basement')
# diagnostic_plots(dataset, 'attic')
# diagnostic_plots(dataset, 'garage')
# diagnostic_plots(dataset, 'price')

data_corr = dataset[continuous_features].corr()
# print(data_corr)

#找到偏移变量，也就是误差值。输出的上下界和上届。
def find_skewed_boundaries(df, variable, distance):
    # 输出该变量的第三分位数
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)  # 输出该变量的第一分位数。

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)  # 这个是下界
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)  # 这个是上界

    return lower_boundary, upper_boundary


# find limits for squareMeters
# There are outliers in squareMeters, floors, basement, attic, garage features.
squareMeters_lower_limit, squareMeters_upper_limit = find_skewed_boundaries(dataset, 'squareMeters', 1.5)
#所以数据小于squareMeters_lower_limit，且大于squareMeters_upper_limit的数值都是异常值。
data = np.where(dataset['squareMeters'] < squareMeters_lower_limit, True, False)
#从这里面已经将小于最大值的数字给取出来了，然后将其赋值为True。
#print(squareMeters_lower_limit, squareMeters_upper_limit)
outliers_squareMeters = np.where(
    dataset['squareMeters'] > squareMeters_upper_limit, True,
    np.where(dataset['squareMeters'] < squareMeters_lower_limit, True, False)
                                 )
# find limits for floors

floors_lower_limit, floors_upper_limit = find_skewed_boundaries(dataset, 'floors', 1.5)
floors_lower_limit, floors_upper_limit
outliers_floors = np.where(dataset['floors'] > floors_upper_limit, True,
                          np.where(dataset['floors'] < floors_lower_limit, True, False))
print(outliers_floors.shape)
basement_lower_limit, basement_upper_limit = find_skewed_boundaries(dataset, 'basement', 1.5)
basement_lower_limit, basement_upper_limit
outliers_basement = np.where(dataset['basement'] > basement_upper_limit, True,
                            np.where(dataset['basement'] < basement_lower_limit, True, False))

#attic
attic_lower_limit,attic_upper_limit = find_skewed_boundaries(dataset,'attic',1.5)
attic_lower_limit,attic_lower_limit
outliers_attic = np.where(dataset['attic'] > attic_upper_limit, True,
                         np.where(dataset['attic'] < attic_lower_limit, True, False))
print(outliers_attic.shape)

garage_lower_limit, garage_upper_limit = find_skewed_boundaries(dataset, 'garage', 1.5)
garage_lower_limit, garage_upper_limit
outliers_garage = np.where(dataset['garage'] > garage_upper_limit, True,
                          np.where(dataset['garage'] < garage_lower_limit, True, False))
print(len(outliers_garage))
for i in range(len(outliers_garage)):
    if outliers_garage[i] == True:
        print(i)

#“~”符号在这里是取反的意思，表示对 df[col].isin(list) 这句返回的值取反，主要用于数据的 slicing。
dataset_trimmed = dataset.loc[~(outliers_squareMeters + outliers_floors + outliers_basement + outliers_attic + outliers_garage), ]
print(dataset.shape, dataset_trimmed.shape)
#删除异常值后，让我们重新调整所有连续特征的诊断图
# diagnostic_plots(dataset_trimmed, 'squareMeters')
# diagnostic_plots(dataset_trimmed, 'floors')
# diagnostic_plots(dataset_trimmed, 'basement')
# diagnostic_plots(dataset_trimmed, 'attic')
# diagnostic_plots(dataset_trimmed, 'garage')
#因此，通过修剪异常值，我们只丢失了 15 个观测值，这不是一个大的数据丢失。但是，其中一些特征不服从高斯分布。
#让我们尝试限制这些异常值而不是修剪（我认为不会有太大变化，在这种情况下，限制异常值对特征分布没有帮助。

#通俗点说，在这里是做了一个修改，将异常值全部修改为了最小值和最大值。
dataset_cap = dataset.copy()
dataset_cap['squareMeters'] = np.where(
    dataset_cap['squareMeters'] > squareMeters_upper_limit,
    squareMeters_upper_limit,
    np.where(dataset_cap['squareMeters'] < squareMeters_lower_limit, squareMeters_lower_limit, dataset_cap['squareMeters'])
                                       )

# floors
dataset_cap['floors'] = np.where(dataset_cap['floors'] > floors_upper_limit, floors_upper_limit,
                          np.where(dataset_cap['floors'] < floors_lower_limit, floors_lower_limit, dataset_cap['floors']))


# basement
dataset_cap['basement'] = np.where(dataset_cap['basement'] > basement_upper_limit, basement_upper_limit,
                            np.where(dataset_cap['basement'] < basement_lower_limit, basement_lower_limit, dataset_cap['basement']))


# attic
dataset_cap['attic'] = np.where(dataset_cap['attic'] > attic_upper_limit, attic_upper_limit,
                         np.where(dataset_cap['attic'] < attic_lower_limit, attic_lower_limit, dataset_cap['attic']))


# garage
dataset_cap['garage'] = np.where(dataset_cap['garage'] > garage_upper_limit, garage_upper_limit,
                          np.where(dataset_cap['garage'] < garage_lower_limit, garage_lower_limit, dataset_cap['garage']))


print(dataset_cap.shape)
# diagnostic_plots(dataset_trimmed, 'squareMeters')
# diagnostic_plots(dataset_trimmed, 'floors')
# diagnostic_plots(dataset_trimmed, 'basement')
# diagnostic_plots(dataset_trimmed, 'attic')
# diagnostic_plots(dataset_trimmed, 'garage')

#正如预期的那样，功能没有变化。事实上，它看起来更糟。所以在下面使用的还是之前的那一批数据，就是删除了15条数据的数据。

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

dataset_trimmed = pd.get_dummies(dataset_trimmed,columns=['cityPartRange', 'hasGuestRoom', 'numPrevOwners'])
#print(dataset_trimmed.columns)

#让我们使用连续和离散特征并构建一个基线模型来了解它是如何出现的。
#print(continuous_features)
required_cols = ['squareMeters', 'numberOfRooms', 'floors', 'basement', 'attic', 'garage']

dataset_use_this = dataset_trimmed[required_cols]
X = dataset_use_this.copy()
y = dataset_trimmed['price']
#print(X.shape,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
params = {'max_depth':[3,6,10],
          'learning_rate':[0.01,0.05,0.1],
          'n_estimators':[100,250,500],
          'colsample_bytree':[0.3,0.7]}
xgbr = xgb.XGBRegressor(seed = 20)
# clf = GridSearchCV(estimator=xgbr,
#                    param_grid=params,
#                    scoring='neg_mean_squared_error',
#                    verbose=1)
# clf.fit(X_train, y_train)
# print("Best parameters:", clf.best_params_)
# print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))

regressor = xgb.XGBRegressor(n_estimators = 500,
                             max_depth = 3,
                             learning_rate = 0.05,
                             colsample_bytree = 1,
                             gamma = 0
                            )
regressor.fit(X_train,y_train)
predicted = regressor.predict(X_test)
print(mean_squared_error(y_test,predicted))

#Test dataset
test_data = val_dataset[required_cols+['id']]
print(test_data.shape)
test_id = test_data['id']
test_dataset_trimmed_1 = test_data.drop(columns='id',axis=1)
print(test_dataset_trimmed_1.shape)
test_predict = regressor.predict(test_dataset_trimmed_1)
print(type(test_id))
test_predict = pd.Series(test_predict)
print(type(test_predict))

