from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline#连续处理操作
from sklearn.preprocessing import StandardScaler#标准化
from sklearn.datasets import make_moons,make_circles,make_classification#数据多样化的建立
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier#需要注意的是版本的对应，这个玩意需要scikit-learn。1.1以上的版本，需要保证python的版本3.9.
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#分类器名称
names = [
    'Nearest Neighbors',
    'Linear SVM',
    'RBF SVN',
    'Decision Tree',
    'Random Forest',
    'Neural Net',
    'AdaBoost',
    'Native Bayes',
]

#分类器实例化
classifier = [
    KNeighborsClassifier(3),
    SVC(kernel='linear',C=0.025),
    SVC(gamma=2,C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5,n_estimators=10,max_features=1),
    MLPClassifier(alpha=1,max_iter=1000),#float，可选，默认为0.0001。L2惩罚（正则化项）参数。
    AdaBoostClassifier(),
    GaussianNB(),
]

#数据的建立
X,y = make_classification(n_features=2,n_redundant=0,n_informative=2,random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)#设置随机变量，以重复多次实验。
X = X + 2*rng.uniform(size=X.shape)#在原有的数据基础之上加上一些噪音。
linearly_separable = (X,y)
datasets = [
    make_moons(noise=0.3,random_state=0),
    make_circles(noise=0.2,factor=0.5,random_state=1),
    linearly_separable,
]

plt.figure(figsize=(27,9))
i = 1

for ds_cnt,ds in enumerate(datasets):
    X,y = ds
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42)

    x_min,x_max = X[:,0].min() - 0.5,X[:,0].max() + 0.5
    y_min,y_max = X[:,1].min() - 0.5,X[:,1].max() + 0.5

    #just plot the dataset first
    cm = plt.cm.RdBu#颜色图
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets),len(classifier)+1,i)
    if ds_cnt == 0:
        ax.set_title('Input data')
    #Plot the training points
    ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap = cm_bright,edgecolors = 'k')
    #Plot the testing points
    ax.scatter(X_test[:,0],X_test[:,1],cmap = cm_bright,edgecolors = 'k',c=y_test,alpha=0.6)

    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i = i + 1

    #iterate over classifier
    for name,clf in zip(names,classifier):

        ax = plt.subplot(len(datasets),len(classifier) + 1,i)

        clf = make_pipeline(StandardScaler(),clf)
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        #Plot the training points
        ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=cm_bright,edgecolors='k')
        #Plot the testing points
        ax.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=cm_bright,edgecolors='k',alpha=0.6)

        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(x_max - 0.3,y_min + 0.3,('%.2f' % score).lstrip('0'),size=15,horizontalalignment='right')
        i = i + 1

plt.tight_layout()
plt.show()