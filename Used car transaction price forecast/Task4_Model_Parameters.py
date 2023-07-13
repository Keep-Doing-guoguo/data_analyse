'''
代码链接：https://tianchi.aliyun.com/notebook/95460
4.1 学习目标
    了解常用的机器学习模型，并掌握机器学习模型的建模与调参流程
    完成相应学习打卡任务
4.2 内容介绍
    1.线性回归模型：
        线性回归对于特征的要求；
        处理长尾分布；
        理解线性回归模型；
    2.模型性能验证：
        评价函数与目标函数；
        交叉验证方法；
        留一验证方法；
        针对时间序列问题的验证；
        绘制学习率曲线；
        绘制验证曲线；
    3.嵌入式特征选择：
        Lasso回归；
        Ridge回归；
        决策树；
    4.模型对比：
        常用线性模型；
        常用非线性模型；
    5.模型调参：
        贪心调参方法；
        网格调参方法；
        贝叶斯调参方法；
4.3 相关原理介绍与推荐
'''

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

sample_feature = reduce_mem_usage(pd.read_csv('data_for_tree.csv'))

continuous_feature_names = [x for x in sample_feature.columns if x not in ['price','brand','model','brand']]#连续性特征变量

sample_feature = sample_feature.dropna().replace('-', 0).reset_index(drop=True)#删除缺失值，然后再替换-为0，重新加一条索引。
sample_feature['notRepairedDamage'] = sample_feature['notRepairedDamage'].astype(np.float32)#转换数据类型
train = sample_feature[continuous_feature_names + ['price']]#新加一个特征

train_X = train[continuous_feature_names]
train_y = train['price']

from sklearn.linear_model import LinearRegression#使用线性回归模型
model = LinearRegression()
model = model.fit(train_X, train_y)

print('intercept:'+ str(model.intercept_))#输出截距的大小
#“斜率”参数（w，也叫作权重或系数）被保存在 coef_ 属性中，而偏移或截距（b）被保存在 intercept_ 属性中
a = sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)#从大到小进行排序

from matplotlib import pyplot as plt

subsample_index = np.random.randint(low=0, high=len(train_y), size=50)

# 画图显示看看我们预测的是否正确，对预测值进行可视化。
# plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
# plt.scatter(train_X['v_9'][subsample_index], model.predict(train_X.loc[subsample_index]), color='blue')
# plt.xlabel('v_9')
# plt.ylabel('price')
# plt.legend(['True Price','Predicted Price'],loc='upper right')
# print('The predicted price is obvious different from true price')
# plt.show()

'''
和条形图的区别，条形图有空隙，直方图没有，条形图一般用于类别特征，直方图一般用于数字特征（连续型）
多用于y值和数字（连续型）特征的分布画图
'''
import seaborn as sns
print('It is clear to see the price shows a typical exponential distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y)
plt.subplot(1,2,2)
sns.distplot(train_y[train_y < np.quantile(train_y, 0.9)])

train_y_ln = np.log(train_y + 1)#将train_y变小一点

import seaborn as sns
print('The transformed price seems like normal distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y_ln)
plt.subplot(1,2,2)
sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])

model = model.fit(train_X, train_y_ln)

print('intercept:'+ str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)


plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], np.exp(model.predict(train_X.loc[subsample_index])), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price seems normal after np.log transforming')
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,  make_scorer

def log_transfer(func):
    def wrapper(y, yhat):
        result = func(np.log(y), np.nan_to_num(np.log(yhat)))
        return result
    return wrapper

scores = cross_val_score(model, X=train_X, y=train_y, verbose=1, cv = 5, scoring=make_scorer(log_transfer(mean_absolute_error)))

print('AVG:', np.mean(scores))

scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=1, cv = 5, scoring=make_scorer(mean_absolute_error))

print('AVG:', np.mean(scores))

scores = pd.DataFrame(scores.reshape(1,-1))
scores.columns = ['cv' + str(x) for x in range(1, 6)]
scores.index = ['MAE']
scores


import datetime
sample_feature = sample_feature.reset_index(drop=True)
split_point = len(sample_feature) // 5 * 4#划分训练集和测试集

train = sample_feature.loc[:split_point].dropna()
val = sample_feature.loc[split_point:].dropna()

train_X = train[continuous_feature_names]
train_y_ln = np.log(train['price'] + 1)
val_X = val[continuous_feature_names]
val_y_ln = np.log(val['price'] + 1)

model = model.fit(train_X, train_y_ln)

mean_absolute_error(val_y_ln, model.predict(val_X))

from sklearn.model_selection import learning_curve, validation_curve
learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_size=np.linspace(.1, 1.0, 5 )):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training example')
    plt.ylabel('score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size, scoring = make_scorer(mean_absolute_error))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()#区域
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label="Training score")
    plt.plot(train_sizes, test_scores_mean,'o-',color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5, n_jobs=1)

train = sample_feature[continuous_feature_names + ['price']].dropna()

train_X = train[continuous_feature_names]
train_y = train['price']
train_y_ln = np.log(train_y + 1)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge#L2正则化操作，L2只会减小特征的权重
from sklearn.linear_model import Lasso#L1正则化操作，L1可以不重要的权重收缩到零

models = [LinearRegression(),
          Ridge(),
          Lasso()]


result = dict()
for model in models:
    model_name = str(model).split('(')[0]#细节大师
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')


result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
result

# model = LinearRegression().fit(train_X, train_y_ln)
# print('intercept:'+ str(model.intercept_))
# sns.barplot(abs(model.coef_), continuous_feature_names)
#
#
# model = Ridge().fit(train_X, train_y_ln)
# print('intercept:'+ str(model.intercept_))
# sns.barplot(abs(model.coef_), continuous_feature_names)
#
#
# model = Lasso().fit(train_X, train_y_ln)
# print('intercept:'+ str(model.intercept_))
# sns.barplot(abs(model.coef_), continuous_feature_names)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor

models = [LinearRegression(),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          MLPRegressor(solver='lbfgs', max_iter=100),
          XGBRegressor(n_estimators = 100, objective='reg:squarederror'),
          LGBMRegressor(n_estimators = 100)]


result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')


result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
result
## 贪心调参
## LGB的参数集合：

objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']

num_leaves = [3,5,10,15,20,40, 55]
max_depth = [3,5,10,15,20,40, 55]
bagging_fraction = []
feature_fraction = []
drop_rate = []

best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(
        cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score

best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0], num_leaves=leaves)
    score = np.mean(
        cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score

best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x: x[1])[0],
                          max_depth=depth)
    score = np.mean(
        cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score

sns.lineplot(x=['0_initial','1_turning_obj','2_turning_leaves','3_turning_depth'], y=[0.143 ,min(best_obj.values()), min(best_leaves.values()), min(best_depth.values())])

from sklearn.model_selection import GridSearchCV
parameters = {'objective': objective , 'num_leaves': num_leaves, 'max_depth': max_depth}
model = LGBMRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(train_X, train_y)
clf.best_params_

model = LGBMRegressor(objective='regression',
                          num_leaves=55,
                          max_depth=15)


np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))

from bayes_opt import BayesianOptimization

def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
    val = cross_val_score(
        LGBMRegressor(objective = 'regression_l1',
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            subsample = subsample,
            min_child_samples = int(min_child_samples)
        ),
        X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)
    ).mean()
    return 1 - val

rf_bo = BayesianOptimization(
    rf_cv,
    {
    'num_leaves': (2, 100),
    'max_depth': (2, 100),
    'subsample': (0.1, 1),
    'min_child_samples' : (2, 100)
    }
)
rf_bo.maximize()

1 - rf_bo.max['target']


plt.figure(figsize=(13,5))
sns.lineplot(x=['0_origin','1_log_transfer','2_L1_&_L2','3_change_model','4_parameter_turning'], y=[1.36 ,0.19, 0.19, 0.14, 0.13])