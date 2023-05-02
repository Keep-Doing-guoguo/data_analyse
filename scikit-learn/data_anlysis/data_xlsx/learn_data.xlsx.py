#该数据集来自吕的玉米光谱数据集。
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay, RocCurveDisplay, log_loss

import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

pd.set_option('display.max_columns', 100) # 设置显示最大列数
pd.set_option('expand_frame_repr', False)
def load_data():
    path = '/Data_an/data.xlsx'
    data = pd.read_excel(path)
    #检查缺失值
    feature_with_na = []
    for feature in data.columns:
        if data[feature].isnull().sum()>1:
            feature_with_na.append(feature)
    for feature in feature_with_na:
        print(feature,np.round(data[feature].isnull().mean(),4),'% missing values')#np.round取整函数,保留4位小数。


    #检查数值型数据类型
    numerical_features = []
    for feature in data.columns:
        if data[feature].dtypes != "0":
            numerical_features.append(feature)
    #print(numerical_features)

    #输出整个数据的类型
    #print(data.dtypes)

    #检查数据的信息
    #print(data.describe().T)

    #输出整个数据信息的
    #print(data.head(5))
    return data
def pca_deal(data):#使用PCA对数据进行降维
    #使用最大似然估计选择超参数
    # n_components='mle' is only supported if n_samples >= n_features
    # pca_mle = PCA(n_components='mle')
    # pca_mle = pca_mle.fit(data)
    # pca_data = pca_mle.transform(data)

    # 按照信息量占比来选择超参数
    # pca_infor = PCA(n_components=0.70,svd_solver='full')
    # pca_infor = pca_infor.fit(data)
    # pca_data = pca_infor.transform(data)
    # print(pca_infor.explained_variance_ratio_)


    # pca = PCA(n_components=20)#选取4个主成分
    # pc = pca.fit_transform(data) #对愿数据进行pca处理
    # print("explained variance ratio: %s" % pca.explained_variance_ratio_) #输出各个主成分所占的比例
    # plt.plot(range(1,21),np.cumsum(pca.explained_variance_ratio_))#绘制主成分累计比例图
    # #plt.scatter(range(1,277),np.cumsum(pca.explained_variance_ratio_))
    # plt.xlim(0,21,1)
    # plt.ylim(0.9,1.02)
    # plt.xlabel("number of components")
    # plt.ylabel("cumulative explained variance")
    # plt.show()


    pca = PCA(n_components=7)#选取7个主成分
    pca_data = pca.fit_transform(data)
    explained_variance_ratio = []
    for x in pca.explained_variance_ratio_:
        data = np.round(x,5)
        explained_variance_ratio.append(data)
    print('explained variance ratio cusum: %s' % np.cumsum(explained_variance_ratio))#从这个得出来结论选择4-7个特征是最好的选择。
    #接下来将会使用网格搜索来寻找最高的准确率
    print("explained variance ratio: %s" % explained_variance_ratio) #输出各个主成分所占的比例

    pca_data_df = pd.DataFrame(pca_data)
    return pca_data_df
def corr(pca_data_df):#画出来相关系数图
    correlation = pca_data_df.corr() #列与列之间的相关系数
    print(correlation)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, annot_kws={'size':16}, cmap='Reds', square=True, ax=ax) #热力图
    plt.show()
def kmeans_deal(pca_data,boole):#进行聚类
    #使用轮廓系数来作为指标的时候。基于轮廓系数来选择n_clusters
    n_clusters = 4
    if boole:
        fig,(ax1,ax2) = plt.subplots(1,2)
        fig.set_size_inches(18,7)
        ax1.set_xlim([-0.1,1])
        ax1.set_ylim([0,pca_data.shape[0] + (n_clusters + 1) * 10])#276 + (4 + 1 )*50
        plt.show()
    clusterer = KMeans(n_clusters=n_clusters,random_state=10).fit(pca_data)
    cluster_labels = clusterer.labels_
    print(cluster_labels)
    silhouette_avg = silhouette_score(pca_data, cluster_labels)
    print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)#平均下来的轮廓系数，单独指的是这一个类别。
    #sample_silhouette_values = silhouette_samples(pca_data, cluster_labels)#指的是这一个类别里面中的每一个样本的轮廓系数。
    #print(sample_silhouette_values)
    pca_data['7'] = cluster_labels
    return pca_data
def watch_data(pca_data_df):#查看数据集标签内容
    data = list(pca_data_df['labels'].value_counts())
    plt.figure(figsize=(6,6))
    plt.bar(x=np.arange(4),height=data,width=0.35)
    plt.xlabel('Labels')
    plt.ylabel('Number')
    plt.show()
def write_data(pca_data_df):
    outputpath = 'E:/ZGW/PycharmProjects1/pythonProject1/Data_an/data.csv'
    pca_data_df.to_csv(outputpath, sep=',', index=False, header=True)
def read_data_csv():
    readpath = 'E:/ZGW/PycharmProjects1/pythonProject1/Data_an/data.csv'
    data = pd.read_csv(readpath)
    return data
    pass
#使用xgboost来进行多分类
def classier(X_train,X_test,bool):
    params ={'learning_rate': 0.1,
              'max_depth': 5,
              'num_boost_round':20,
              'objective': 'multi:softmax',
              'random_state': 27,
              'silent':0,
              'num_class':4
            }
    model = xgb.XGBClassifier(objective= 'multi:softmax',random_state=42)
    result = model.fit(X_train.iloc[:,:-1], X_train.iloc[:,-1])
    pred = model.predict(X_test.iloc[:,:-1])
    acc = accuracy_score(y_true=X_test.iloc[:,-1],y_pred=pred)
    mat = confusion_matrix(y_true=X_test.iloc[:,-1],y_pred=pred)
    if bool:
        plt.figure(figsize=(6,6))
        sns.heatmap(data=mat,cmap='Blues',fmt='g',annot=True)
        plt.title('Confusion Max')
        plt.xlabel('Predcited')
        plt.ylabel('Actual')
        plt.show()
    print(acc)

#使用决策树来进行多分类
def descision_tree(X_train,X_test,bool):
    model = DecisionTreeClassifier(criterion='entropy',splitter='random',random_state=42)#保证结果的随机可验证性
    model = model.fit(X_train.iloc[:,:-1],X_train.iloc[:,-1])
    pred = model.predict(X_test.iloc[:,:-1])
    y_true = np.array(X_test.iloc[:,-1])
    acc_score = accuracy_score(y_true=y_true,y_pred=pred)
    print('acc_score: ', acc_score)
    mat = confusion_matrix(y_true=y_true,y_pred=pred)
    if bool:
        plt.figure(figsize=(6,6))
        sns.heatmap(data=mat,annot=True,cmap='Blues',fmt='g')
        plt.xlabel('T_Pred')
        plt.ylabel('Y_Pred')
        plt.title('Confusion Max')
        plt.show()
    print(model.feature_importances_)#输出特征的重要性
#使用随机森林来进行多分类。
def random_classier_tree(X_train,X_test,bool):
    model = RandomForestClassifier(random_state=42)
    model = model.fit(X_train.iloc[:,:-1],X_train.iloc[:,-1])
    y_pred = model.predict(X_test.iloc[:,:-1])
    y_true = np.array(X_test.iloc[:, -1])
    acc_score = accuracy_score(y_true=y_true,y_pred=y_pred)
    mat = confusion_matrix(y_true=y_true,y_pred=y_pred)
    if bool:
        plt.figure(figsize=(6, 6))
        sns.heatmap(data=mat, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('T_Pred')
        plt.ylabel('Y_Pred')
        plt.title('Confusion Max')
        plt.show()
    print('acc_score:',acc_score)
    pass
#使用支持向量机来进行多分类
def svm_classier(X_train,X_test,bool):

    pass
def use_gridsearch(classier,X_train,**kwargs):
    grid = GridSearchCV(classier,param_grid=kwargs,return_train_score=True)
    with tqdm(total=len(kwargs),desc='Hyperparameter tuning') as pbar:
        grid_result = grid.fit(X=X_train.iloc[:,:-1],y=X_train.iloc[:,-1])
        pbar.update()

if __name__ == '__main__':
    # data = load_data()
    # pca_data_df = pca_deal(data)
    # pca_data_df = kmeans_deal(pca_data_df,False)
    # print(pca_data_df.head())
    # write_data(pca_data_df)

    params_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.1, 0.5],
        'subsample': [0.5, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5],

        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 1, 10]
    }
    data = read_data_csv()
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    ##############################1.决策树##############################
    #descision_tree(X_train,X_test,True)
    ##############################1.决策树##############################

    ##############################2.随机森立##############################
    random_classier_tree(X_train,X_test,True)
    ##############################2.随机森立##############################

    ##############################2.随机森立##############################


    ##############################3.xgboost##############################
    # model = xgb.XGBClassifier(objective='multi:softmax', random_state=42)
    # use_gridsearch(classier=model,X_train=data,**params_grid)
    ##############################3.xgboost##############################
#创建模型。


