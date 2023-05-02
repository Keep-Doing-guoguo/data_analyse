import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix
from math import sqrt
import xgboost as xgb

pd.set_option('display.max_rows', 200)
pd.set_option('expand_frame_repr', False)

test_path = 'E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e12/test.csv'
train_path = 'E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e12/train.csv'
original_path = 'E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e12/kindey stone urine analysis.csv'

test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)
original_data = pd.read_csv(original_path)

def load_data(test_data,train_data,original_data):
    print(test_data.shape)
    print(train_data.shape)
    print(original_data.shape)

    print(test_data.info())
    print(train_data.info())
    print(original_data.info())

    print(test_data.describe().T)
    print(train_data.describe().T)
    print(original_data.describe().T)

    for i in train_data.columns:
        print(train_data[i].nunique())
    print()

def deal_outlier(df, variable, distance):
    distance = 1.5
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundaty = df[variable].quantile(0.75) + (IQR * distance)

    return lower_boundary,upper_boundaty
def analyse_data(test_data,train_data,original_data):
    test_data = test_data.drop('id', axis=1)
    train_data = train_data.drop('id', axis=1)
    print()
    outiler_features = ['ph', 'gravity']
    # for i in outiler_features:
    #     plt.figure()
    #     sns.boxplot(train_data[i])
    # plt.show()
    #deal with train data
    lower_boundary, upper_boundaty = deal_outlier(train_data,'ph',1.5)
    train_data = train_data[
        (train_data['ph'] > lower_boundary) & (train_data['ph'] < upper_boundaty)
    ]

    lower_boundary, upper_boundaty = deal_outlier(train_data, 'gravity', 1.5)
    train_data = train_data[
        (train_data['gravity'] > lower_boundary) & (train_data['gravity'] < upper_boundaty)
    ]
    print(train_data.shape)

    y_label = train_data.iloc[:,-1]
    print(y_label.head())
    #deal with test data
    lower_boundary, upper_boundaty = deal_outlier(test_data, 'ph', 1.5)
    test_data = test_data[
        (test_data['ph'] > lower_boundary) & (test_data['ph'] < upper_boundaty)
    ]
    lower_boundary, upper_boundaty = deal_outlier(test_data, 'gravity', 1.5)
    test_data = test_data[
        (test_data['gravity'] > lower_boundary) & (test_data['gravity'] < upper_boundaty)
    ]
    # for i in outiler_features:
    #     plt.figure()
    #     sns.boxplot(train_data[i])
    # plt.show()
    # data preprocessing
    sd = StandardScaler()
    print(train_data.columns)
    train_data = pd.DataFrame(sd.fit_transform(train_data.iloc[:,:6]))
    return train_data,y_label,test_data

def plot_image(test_data,train_data,original_data):
    # drop id
    test_data = test_data.drop('id', axis=1)
    train_data = train_data.drop('id', axis=1)
    y_label = train_data.iloc[:, -1].values  # 转换为列表

    # #relation correlation
    # plt.figure(figsize=(8,8))
    # corr = train_data.corr()
    # sns.heatmap(corr,annot=True, annot_kws={'size':16}, cmap='Blues', square=True)
    # plt.xticks(range(len(train_data.columns)),train_data.columns)
    #
    #
    #plot various images
    #sns.pairplot(train_data,hue='target')


    # #plot histplot and boxplot
    numerical_features = ['gravity','ph','osmo','cond','urea','calc']
    # for i in numerical_features:
    #     plt.figure(figsize=(12,8))
    #     plt.subplot(1,2,1)
    #     sns.histplot(train_data[i],bins=30,kde=True)
    #
    #     plt.subplot(1,2,2)
    #     sns.boxplot(train_data[i])
    # #we can conclude the ph and gravity has outlier

    # #plot number of target
    # plt.figure(figsize=(8,8))
    # list_traget = train_data['target'].value_counts().tolist()
    # color = ['blue','violet']
    # plt.bar(x=[0,1],height=list_traget,color=color)
    # plt.legend()
    # plt.xticks(range(0,2),['0.0','1.0'])
    # plt.show()
    # print()
    #
    # plt.show()

    pass

def RFC(test_data,train_data,original_data):
    train_data, y_label = analyse_data(test_data,train_data,original_data)
    rfc = RandomForestClassifier()
    rfc = rfc.fit(train_data,y_label)
    print()


    pass

def DTC(test_data,train_data,original_data):
    train_data, y_label ,test_data= analyse_data(test_data, train_data, original_data)
    model = DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=42)  # 保证结果的随机可验证性
    model = model.fit(train_data, y_label)
    pred = model.predict(train_data)

    acc_score = accuracy_score(y_true=y_label, y_pred=pred)
    print('acc_score: ', acc_score)
    mat = confusion_matrix(y_true=y_label, y_pred=pred)
    if bool:
        plt.figure(figsize=(6,6))
        sns.heatmap(data=mat,annot=True,cmap='Blues',fmt='g')
        plt.xlabel('T_Pred')
        plt.ylabel('Y_Pred')
        plt.title('Confusion Max')
        plt.show()
    print(model.feature_importances_)#输出特征的重要性
if  __name__ == '__main__':
    #load_data(test_data, train_data, original_data)
    # plot_image(test_data,train_data,original_data)
    # analyse_data(test_data, train_data, original_data)
    DTC(test_data, train_data, original_data)