## importing packages
import argparse
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression#互信息回归
import pandas as pd
pd.set_option('display.max_rows', 200)
pd.set_option('expand_frame_repr', False)

##2.read data
def read_data(args):
    train_path = os.path.join(args.data_path,'train.csv','train.csv')
    test_path = os.path.join(args.data_path,'test.csv','test.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    actual = pd.read_csv(os.path.join(args.data_path,'cubic_zirconia.csv'))
    # print(train.head())
    # print(test.head())
    #删除id列
    train = train.drop('id',axis=1)
    test = test.drop('id',axis=1)
    #检查actual数据集是否有null值
    actual_info = actual.info()
    actual = actual.dropna(axis=0)#将这一行其中有null的数据给删除了。对于少部分有null值的可以如此处理。
    actual = actual.drop('Unnamed: 0',axis=1)
    # print(actual_info)
    return train,test,actual
##Concating data sets拼接数据集
def concate_data(args):
    train, test, actual = read_data(args)
    og = pd.concat([train,actual])
    return og,train,test,actual
##EDA
def EDA_data(args):
    #
    og,train,test,actual = concate_data(args)
    print(og.info())
    print(og.describe().T)
    #correlation matirx
    all_features = og.columns
    numeric_feature = og[['carat' , 'depth' ,'table' ,'x' ,'y' ,'z' , 'price']]
    # plt.figure(figsize=(8,8))
    # sns.heatmap(numeric_feature.corr(),annot=True,fmt='g')
    # plt.show()

    # plt.figure(figsize=(8,8))
    # #斯皮尔曼相关系数
    # #https://blog.csdn.net/chenxy_bwave/article/details/121427036
    # sns.heatmap(numeric_feature.corr('spearman'),annot=True)
    # plt.show()

    #correlation matrix
    categorical_feature = og[['cut','color','clarity']]
    encoder = LabelEncoder()#主要用于标签
    encoded = categorical_feature.apply(encoder.fit_transform)
    encoded['price'] = train['price']
    #如果要想3列同时进行，就需要像上面的代码一样。
    # encoded = encoder.fit_transform(categorical_feature)
    # encoded['price'] = train['price']

    # plt.figure(figsize=(8,8))
    # sns.heatmap(encoded.corr('kendall'),annot=True,fmt='g',cmap='Blues')
    # plt.show()
    return og,train,test,actual

    print()

def plots_data(args):
    og, train, test, actual = EDA_data(args)
    plots = True
    numerical_features = og[['carat' , 'depth' ,'table' ,'x' ,'y' ,'z' , 'price']]
    obecjt_features_data = og[['cut', 'color', 'clarity']]
    #在这里只画float数据类型的数据值。
    if plots:

        for i in numerical_features:
            fig, ax = plt.subplots(1, 5, figsize=(24, 4))
            #histplot绘制直方图,将核密度曲线画出来。
            sns.histplot(og[i],bins=30,kde=True,ax=ax[0])
            ax[0].set_title('Histtogram')
            #KDE plot
            sns.kdeplot(og[i],ax=ax[1])
            ax[1].set_title('KDE Plot')
            #Q-Q plots
            stats.probplot(og[i],dist='norm',plot=ax[2])
            ax[2].set_title('Q-Q Plot')
            #boxplot
            sns.boxplot(y=og[i],ax=ax[3])
            ax[3].set_title('Boxplot')
            #scatterplot,data.index指的是data的索引
            sns.scatterplot(x=og.index,y=og[i],ax=ax[4])
            ax[4].set_title('Scatterplot')
            plt.tight_layout()
            plt.show()
    #在这里只画object数据类型的数据
    plots_object = False

    if plots_object:
        fig,ax = plt.subplots(1,3,figsize=(20,4))
        for i,col in enumerate(obecjt_features_data.culumns):
            sns.countplot(x=col,data=obecjt_features_data,ax=ax[i])
            ax[i].set_title(col)

        plt.tight_layout()
        plt.show()
def process(df):
    # Handling Categorical variables
    color_dic = {'D': 6, 'E': 5, 'F': 4, 'G': 3, 'H': 2, 'I': 1, 'J': 0}
    clarity_dic = {'FL': 10, 'IF': 9, 'VVS1': 8, 'VVS2': 7, 'VS1': 6, 'VS2': 5, 'SI1': 4, 'SI2': 3, 'I1': 2, 'I2': 1,
                   'I3': 0}

    df['color'] = df['color'].apply(lambda x:color_dic[x])
    df['clarity'] = df['clarity'].apply(lambda x:clarity_dic[x])
    return df
#Outliers info删除一些异常值。
def detect_outliers(args):
    og, train, test, actual = EDA_data(args)
    numerical_features_data = og[['carat', 'depth', 'table', 'x', 'y', 'z', 'price']]
    obecjt_features_data = og[['cut', 'color', 'clarity']]
    outlier_percents = {}
    for column in numerical_features_data.columns:
        #分别是0.25，0.5，0.75
        #计算给定数据（数组元素）沿指定轴线的第q个四分位数
        print(column)
        q1 = np.quantile(numerical_features_data[column],0.25)
        q3 = np.quantile(numerical_features_data[column],0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q1 + 1.5 * iqr
        outlier = numerical_features_data[
            (numerical_features_data[column] < lower_bound)|(numerical_features_data[column]>upper_bound)
        ]
        outlier_percent = (outlier.shape[0] / numerical_features_data.shape[0]) * 100
        outlier_percents[column] = outlier_percent

    outlier_dataframe = pd.DataFrame(data=outlier_percents.values(),index=outlier_percents.keys(),columns=['outlier_percents'])
    print(outlier_dataframe.sort_values(by = 'outlier_percents'))
    # feature importance
    vif_list = []
    for i in range(len(numerical_features_data.columns)):
        vif = variance_inflation_factor(numerical_features_data.to_numpy(),i)#多重共线性检验
        vif_list.append(vif)

    s1 = pd.Series(data=vif_list,index=numerical_features_data.columns)
    # s1.sort_values().plot(kind = 'barh')
    # plt.show()

    fea = numerical_features_data.drop('price' , axis=1)
    tar = numerical_features_data['price']
    # array = mutual_info_regression(fea,tar)
    # series = pd.Series(array, index=fea.columns)
    # series.sort_values().plot(kind='barh')
    # plt.show()

    ##model preparing
    #Handling
    og_color_unique = og['color'].unique()
    og_clarity_unique = og['clarity'].unique()
    og_cut_unique = og.cut.unique()
    color_dic = {'D': 6, 'E': 5, 'F': 4, 'G': 3, 'H': 2, 'I': 1, 'J': 0}
    clarity_dic = {'FL': 10, 'IF': 9, 'VVS1': 8, 'VVS2': 7, 'VS1': 6, 'VS2': 5, 'SI1': 4, 'SI2': 3, 'I1': 2, 'I2': 1,
                   'I3': 0}

    prediction_array = {'Premium': 0,
                        'Very Good': 0,
                        'Ideal': 0,
                        'Good': 0,
                        'Fair': 0}

    skf = StratifiedKFold(shuffle=True, random_state=42)
    for num,i in enumerate(og.cut.unique()):
        df = og.copy()
        test = test.copy()

        data_ = df[df['cut'] == i].drop('cut' , axis=1)
        test_str_ = test[test['cut'] == i].drop(['cut'], axis=1)

        data = process(data_)
        test_str = process(test_str_)

        X = data.drop('price', axis=1)
        y = data['price']

        y_preds = []
        for fold,(train_idx,test_idx) in enumerate(skf.split(X,y)):
            X_train,X_test = X.iloc[train_idx],X.iloc[test_idx]
            y_train,y_test = y.iloc[train_idx],y.iloc[test_idx]

            model = XGBRegressor()

            model.fit(X_train,y_train)

            y_pred = model.predict(test_str)
            y_preds.append(y_pred)#5折交叉验证会出来5次的预测结果信息
        sample = pd.DataFrame(data=y_preds)
        sample = sample.T
        sample.columns = ['1','2','3','4','5']
        sample['add'] = sample.apply(lambda x :((x['1']+x["2"]+x["3"]+x["4"]+x["5"])/5) ,axis = 1)
        x = np.array(sample['add'])
        prediction_array[i] = x
        print()
    print('ok')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',type=str,default='E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e8')
    args = parser.parse_args()
    detect_outliers(args)