from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston#用于回归的数据集
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime


#加载数据集信息
data = load_boston()
#波士顿数据集非常简单，但它所涉及到的问题却很多
X = data.data
y = data.target
print(X.shape)
print(y.shape)

#划分数据集信息
Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)
reg = XGBR(n_estimators=100).fit(Xtrain,Ytrain) #训练
reg.predict(Xtest) #传统接口predict
R2 = reg.score(Xtest,Ytest) #你能想出这里应该返回什么模型评估指标么？利用shift+Tab可以知道，R^2评估指标
print('R2',R2)
print(y.mean())#输出y的平均值，
MSE_score = MSE(Ytest,reg.predict(Xtest))#可以看出均方误差是平均值y.mean()的1/3左右，结果不算好也不算坏
print('MSE_score',MSE_score)
#可以查看到模型特征的贡献数
feature_im = reg.feature_importances_ #树模型的优势之一：能够查看模型的重要性分数，可以使用嵌入法(SelectFromModel)进行特征选择
#xgboost可以使用嵌入法进行特征选择
reg = XGBR(n_estimators=100) #交叉验证中导入的没有经过训练的模型
cvs_socre = CVS(reg,Xtrain,Ytrain,cv=5).mean()
#这里应该返回什么模型评估指标，还记得么？ 返回的是与reg.score相同的评估指标R^2（回归），准确率（分类）
print('cvs_socre',cvs_socre)
cvs_mse = CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()

#查看sklearn总所有的模型评估指标
import sklearn
print(sorted(sklearn.metrics.SCORERS.keys()))

#使用随机森林和线性回归进行一个对比
rfr = RFR(n_estimators=100)#随机森林模型
cvs_socre_r2 = CVS(rfr,Xtrain,Ytrain,cv=5).mean()#0.7975497480638329
CVS(rfr,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()#-16.998723616338033

lr = LinearR()#线性回归模型
CVS(lr,Xtrain,Ytrain,cv=5).mean()#0.6835070597278085
CVS(lr,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()#-25.34950749364844

#如果开启参数slient：在数据巨大，预料到算法运行会非常缓慢的时候可以使用这个参数来监控模型的训练进度
reg = XGBR(n_estimators=10,silent=True)#xgboost库silent=True不会打印训练进程，只返回运行结果，默认是False会打印训练进程
#sklearn库中的xgbsoost的默认为silent=True不会打印训练进程，想打印需要手动设置为False
CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()#-92.67865836936579


def plot_learning_curve(estimator, title, X, y,
                        ax=None,  # 选择子图
                        ylim=None,  # 设置纵坐标的取值范围
                        cv=None,  # 交叉验证
                        n_jobs=None  # 设定索要使用的线程
                        ):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                            , shuffle=True
                                                            , cv=cv
                                                            , random_state=420
                                                            , n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()  # 绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-'
            , color="r", label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-'
            , color="g", label="Test score")
    ax.legend(loc="best")
    return ax


cv = KFold(n_splits=5, shuffle = True, random_state=42) #交叉验证模式
plot_learning_curve(XGBR(n_estimators=100,random_state=420)
                    ,"XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()