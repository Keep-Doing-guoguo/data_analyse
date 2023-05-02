import pandas as pd
import numpy as np
path = 'C:/Users/123/PycharmProjects/pythonProject/wenwen/scikit-learn/聚类实践/data.xlsx'
data = pd.read_excel(path)
#data = data.iloc[1:,:]
#print(data.head(5))
print(data.shape)
print(type(data))
#检查缺失值
feature_with_na = []
for feature in data.columns:
    if data[feature].isnull().sum()>1:
        feature_with_na.append(feature)
for feature in feature_with_na:
    print(feature,np.round(data[feature].isnull().mean(),4),'% missing values')#np.round取整函数,保留4位小数。
import sklearn.metrics


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

#使用PCA对数据进行降维

#使用最大似然估计选择超参数
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# n_components='mle' is only supported if n_samples >= n_features
# pca_mle = PCA(n_components='mle')
# pca_mle = pca_mle.fit(data)
# pca_data = pca_mle.transform(data)

# 按照信息量占比来选择超参数
# pca_infor = PCA(n_components=0.70,svd_solver='full')
# pca_infor = pca_infor.fit(data)
# pca_data = pca_infor.transform(data)
# print(pca_infor.explained_variance_ratio_)

# pca = PCA(n_components=276)#选取4个主成分
# pc = pca.fit_transform(data) #对愿数据进行pca处理
# print("explained variance ratio: %s" % pca.explained_variance_ratio_) #输出各个主成分所占的比例
# plt.plot(range(1,277),np.cumsum(pca.explained_variance_ratio_))#绘制主成分累计比例图
# #plt.scatter(range(1,277),np.cumsum(pca.explained_variance_ratio_))
# plt.xlim(0,277,1)
# plt.ylim(0.9,1.02)
# plt.xlabel("number of components")
# plt.ylabel("cumulative explained variance")
# plt.show()


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
print("explained variance ratio: %s" % pca.explained_variance_ratio_) #输出各个主成分所占的比例
import seaborn as sns
pca_data_df = pd.DataFrame(pca_data)
correlation = pca_data_df.corr() #列与列之间的相关系数
print(correlation)
fig, ax = plt.subplots(figsize=(12, 10))
#sns.heatmap(correlation, annot=True, annot_kws={'size':16}, cmap='Reds', square=True, ax=ax) #热力图
sns.pairplot(data) #散点关系图
plt.show()

#得到的新特征的信息
# print(pca_data_df.head(5))
# print(pca_data.shape[0])

#进行聚类
from sklearn.cluster import KMeans
#使用轮廓系数来作为指标的时候。基于轮廓系数来选择n_clusters
from sklearn.metrics import silhouette_score,silhouette_samples
n_clusters = 4
fig,(ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(18,7)
ax1.set_xlim([-0.1,1])
ax1.set_ylim([0,pca_data.shape[0] + (n_clusters + 1) * 10])#276 + (4 + 1 )*50
clusterer = KMeans(n_clusters=n_clusters,random_state=10).fit(pca_data)
cluster_labels = clusterer.labels_
silhouette_avg = silhouette_score(pca_data, cluster_labels)
print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)#平均下来的轮廓系数，单独指的是这一个类别。
sample_silhouette_values = silhouette_samples(pca_data, cluster_labels)#指的是这一个类别里面中的每一个样本
y_lower = 10
import matplotlib.cm as cm #colormap
for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)

    ax1.fill_betweenx(np.arange(y_lower, y_upper)
                      , ith_cluster_silhouette_values
                      , facecolor=color
                      , alpha=0.7
                      )

    ax1.text(-0.05
             , y_lower + 0.5 * size_cluster_i
             , str(i))

    y_lower = y_upper + 10

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])

ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


#使用卡林斯基-哈拉巴斯指数作为评估指标，时间快。
#from sklearn.metrics import calinski_harabaz_score

