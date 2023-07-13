'''
代码链接：
https://tianchi.aliyun.com/notebook/95457
2.1 EDA目标
    EDA的价值主要在于熟悉数据集，了解数据集，对数据集进行验证来确定所获得数据集可以用于接下来的机器学习或者深度学习使用。

    当了解了数据集之后我们下一步就是要去了解变量间的相互关系以及变量与预测值之间的存在关系。

    引导数据科学从业者进行数据处理以及特征工程的步骤,使数据集的结构和特征集让接下来的预测问题更加可靠。

    完成对于数据的探索性分析，并对于数据进行一些图表或者文字总结并打卡。

2.2 内容介绍
    1.载入各种数据科学以及可视化库:
        数据科学库 pandas、numpy、scipy；
        可视化库 matplotlib、seabon；
        其他；
    2.载入数据：
        载入训练集和测试集；
        简略观察数据(head()+shape)；
    3.数据总览:
        通过describe()来熟悉数据的相关统计量
        通过info()来熟悉数据类型
    4.判断数据缺失和异常
        查看每列的存在nan情况
        异常值检测
    5.了解预测值的分布
        总体分布概况（无界约翰逊分布等）
        查看skewness and kurtosis
        查看预测值的具体频数
    6.特征分为类别特征和数字特征，并对类别特征查看unique分布
    7.数字特征分析
        相关性分析
        查看几个特征得 偏度和峰值
        每个数字特征得分布可视化
        数字特征相互之间的关系可视化
        多变量互相回归关系可视化
    8.类型特征分析
        unique分布
        类别特征箱形图可视化
        类别特征的小提琴图可视化
        类别特征的柱形图可视化类别
        特征的每个类别频数可视化(count_plot)
    9.用pandas_profiling生成数据报告

'''


#coding:utf-8
#导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno#missingno 是一个可以将缺失值情况进行可视化的库


## 1) 载入训练集和测试集；
train_path = '/Data_an/Used car transaction price forecast/used_car_train_20200313.csv'

test_path = '/Data_an/Used car transaction price forecast/used_car_testB_20200421.csv'

Train_data = pd.read_csv(train_path, sep=' ')
Test_data = pd.read_csv(test_path, sep=' ')

## 2) 简略观察数据(head()+shape)
a = Train_data.head().append(Train_data.tail())#观察数据的尾部信息
b = Test_data.head().append(Test_data.tail())

## 1) 通过describe()来熟悉数据的相关统计量
Train_data.describe()
Test_data.describe()

## 2) 通过info()来熟悉数据类型
Train_data.info()
Test_data.info()


## 1) 查看每列的存在nan情况
Train_data.isnull().sum()
Test_data.isnull().sum()
# nan可视化
missing = Train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
# 可视化看下缺省值
#msno.matrix(Train_data.sample(250))#从Train_data中取出来250个样本，其中总共有31条数据，其中28条是完整没有缺失值的。
msno.bar(Train_data.sample(1000))
# 可视化看下缺省值，对测试集进行画图显示。
# msno.matrix(Test_data.sample(250))
# msno.bar(Test_data.sample(1000))
## 2) 查看异常值检测

c = Train_data['notRepairedDamage'].value_counts()#对这个类别进行计数，看看其中的每一个类别有多少个。
c1 = Train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)#进行替换操作
c2 = Train_data['notRepairedDamage'].value_counts()#再次计数查看
c3 = Train_data.isnull().sum()
c4 = Test_data['notRepairedDamage'].value_counts()
c5 = Test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
c6 = Train_data["seller"].value_counts()
c7 = Train_data["offerType"].value_counts()#类别个数差别太大，所以进行删除。
del Train_data["seller"]
del Train_data["offerType"]
del Test_data["seller"]
del Test_data["offerType"]


Train_data['price']
c8 = Train_data['price'].value_counts()#能够观察到价格的分布信息
## 1) 总体分布概况（无界约翰逊分布等）
import scipy.stats as st
y = Train_data['price']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
plt.figure(4); plt.title('Normal Normal')
sns.distplot(y)
## 2) 查看skewness and kurtosis
sns.distplot(Train_data['price']);
print("Skewness: %f" % Train_data['price'].skew())#某总体取值分布的对称性，简单来说就是数据的不对称程度。。
print("Kurtosis: %f" % Train_data['price'].kurt())#取值分布形态陡缓程度的统计量，简单来说就是数据分布顶的尖锐程度。
Train_data.skew(), Train_data.kurt()
plt.figure(6)
sns.distplot(Train_data.skew(),color='blue',axlabel ='Skewness')
plt.figure(7)
sns.distplot(Train_data.kurt(),color='orange',axlabel ='Kurtness')
## 3) 查看预测值的具体频数
plt.hist(Train_data['price'], orientation = 'vertical',histtype = 'bar', color ='red')
# plt.show()
# log变换 z之后的分布较均匀，可以进行log变换进行预测，这也是预测问题常用的trick
plt.hist(np.log(Train_data['price']), orientation = 'vertical',histtype = 'bar', color ='red')#直方图，mataplotlib
# plt.show()

# 分离label即预测值
Y_train = Train_data['price']
# 这个区别方式适用于没有直接label coding的数据
# 这里不适用，需要人为根据实际含义来区分
# 数字特征
# numeric_features = Train_data.select_dtypes(include=[np.number])
# numeric_features.columns
# # 类型特征
# categorical_features = Train_data.select_dtypes(include=[np.object])
# categorical_features.columns
numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]

categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode',]

# 特征nunique分布
for cat_fea in categorical_features:
    print(cat_fea + "的特征分布如下：")
    print("{}特征有个{}不同的值".format(cat_fea, Train_data[cat_fea].nunique()))
    print(Train_data[cat_fea].value_counts())


# 特征nunique分布
for cat_fea in categorical_features:
    print(cat_fea + "的特征分布如下：")
    print("{}特征有个{}不同的值".format(cat_fea, Test_data[cat_fea].nunique()))
    print(Test_data[cat_fea].value_counts())


numeric_features.append('price')
d = numeric_features

## 1) 相关性分析
price_numeric = Train_data[numeric_features]
correlation = price_numeric.corr()
print(correlation['price'].sort_values(ascending = False),'\n')
f , ax = plt.subplots(figsize = (7, 7))

plt.title('Correlation of Numeric Features with Price',y=1,size=16)

sns.heatmap(correlation,square = True,  vmax=0.8)
del price_numeric['price']
## 2) 查看几个特征得 偏度和峰值
for col in numeric_features:
    print('{:15}'.format(col),
          'Skewness: {:05.2f}'.format(Train_data[col].skew()) ,
          '   ' ,
          'Kurtosis: {:06.2f}'.format(Train_data[col].kurt())
         )
## 3) 每个数字特征得分布可视化
# f = pd.melt(Train_data, value_vars=numeric_features)
# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
# g = g.map(sns.distplot, "value")

## 4) 数字特征相互之间的关系可视化
sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
# plt.show()

Train_data.columns

Y_train

## 5) 多变量互相回归关系可视化
# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(24, 20))
# # ['v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
# v_12_scatter_plot = pd.concat([Y_train,Train_data['v_12']],axis = 1)
# sns.regplot(x='v_12',y = 'price', data = v_12_scatter_plot,scatter= True, fit_reg=True, ax=ax1)
#
# v_8_scatter_plot = pd.concat([Y_train,Train_data['v_8']],axis = 1)
# sns.regplot(x='v_8',y = 'price',data = v_8_scatter_plot,scatter= True, fit_reg=True, ax=ax2)
#
# v_0_scatter_plot = pd.concat([Y_train,Train_data['v_0']],axis = 1)
# sns.regplot(x='v_0',y = 'price',data = v_0_scatter_plot,scatter= True, fit_reg=True, ax=ax3)
#
# power_scatter_plot = pd.concat([Y_train,Train_data['power']],axis = 1)
# sns.regplot(x='power',y = 'price',data = power_scatter_plot,scatter= True, fit_reg=True, ax=ax4)
#
# v_5_scatter_plot = pd.concat([Y_train,Train_data['v_5']],axis = 1)
# sns.regplot(x='v_5',y = 'price',data = v_5_scatter_plot,scatter= True, fit_reg=True, ax=ax5)
#
# v_2_scatter_plot = pd.concat([Y_train,Train_data['v_2']],axis = 1)
# sns.regplot(x='v_2',y = 'price',data = v_2_scatter_plot,scatter= True, fit_reg=True, ax=ax6)
#
# v_6_scatter_plot = pd.concat([Y_train,Train_data['v_6']],axis = 1)
# sns.regplot(x='v_6',y = 'price',data = v_6_scatter_plot,scatter= True, fit_reg=True, ax=ax7)
#
# v_1_scatter_plot = pd.concat([Y_train,Train_data['v_1']],axis = 1)
# sns.regplot(x='v_1',y = 'price',data = v_1_scatter_plot,scatter= True, fit_reg=True, ax=ax8)
#
# v_14_scatter_plot = pd.concat([Y_train,Train_data['v_14']],axis = 1)
# sns.regplot(x='v_14',y = 'price',data = v_14_scatter_plot,scatter= True, fit_reg=True, ax=ax9)
#
# v_13_scatter_plot = pd.concat([Y_train,Train_data['v_13']],axis = 1)
# sns.regplot(x='v_13',y = 'price',data = v_13_scatter_plot,scatter= True, fit_reg=True, ax=ax10)
## 1) unique分布
for fea in categorical_features:
    print(Train_data[fea].nunique())



## 2) 类别特征箱形图可视化

# 因为 name和 regionCode的类别太稀疏了，这里我们把不稀疏的几类画一下
categorical_features = ['model',
 'brand',
 'bodyType',
 'fuelType',
 'gearbox',
 'notRepairedDamage']
for c in categorical_features:
    Train_data[c] = Train_data[c].astype('category')
    if Train_data[c].isnull().any():
        Train_data[c] = Train_data[c].cat.add_categories(['MISSING'])
        Train_data[c] = Train_data[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "price")

Train_data.columns

## 3) 类别特征的小提琴图可视化
catg_list = categorical_features
target = 'price'
for catg in catg_list :
    sns.violinplot(x=catg, y=target, data=Train_data)
    plt.show()

categorical_features = ['model',
 'brand',
 'bodyType',
 'fuelType',
 'gearbox',
 'notRepairedDamage']

## 4) 类别特征的柱形图可视化
def bar_plot(x, y, **kwargs):
    sns.barplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(bar_plot, "value", "price")


##  5) 类别特征的每个类别频数可视化(count_plot)
def count_plot(x,  **kwargs):
    sns.countplot(x=x)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data,  value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(count_plot, "value")
