#https://www.kaggle.com/competitions/playground-series-s3e8/data
'''
 Carat weight of the cubic zirconia.
 Describe the cut quality of the cubic zirconia. Quality is increasing order Fair, Good, Very Good, Premium, Ideal.
 Colour of the cubic zirconia.With D being the best and J the worst.
 cubic zirconia Clarity refers to the absence of the Inclusions and Blemishes.
(In order from Best to Worst, FL = flawless, I3= level 3 inclusions) FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3
 The Height of a cubic zirconia, measured from the Culet to the table, divided by its average Girdle Diameter.
 The Width of the cubic zirconia's Table expressed as a Percentage of its Average Diameter.
 Length of the cubic zirconia in mm.
 Width of the cubic zirconia in mm.
 Height of the cubic zirconia in mm.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

#依次分别为单一数字，独热向量，标签专用
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split,KFold,cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb
import xgboost as xgb
pd.set_option('display.max_rows', 200)
pd.set_option('expand_frame_repr', False)
def read_data(args):
    train_path = os.path.join(args.data_path,'train.csv','train.csv')
    test_path = os.path.join(args.data_path,'test.csv','test.csv')
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    #需要对多出来的id这一项进行删除
    df_train = df_train.drop('id',axis=1)#0代表的是行，1代表的是列。
    df_test = df_test.drop('id',axis=1)
    # print(df_train.head())
    # print(df_test.head())
    # print(df_train.info())
    # print(df_test.info())
    # print(df_train.describe().T)
    # print(df_test.describe().T)
    return df_train,df_test
def process_data(args):
    df_train, df_test = read_data(args)
    #Encode string values to integers
    #For Cut Varible
    cut_unique = df_train['cut'].unique()
    cut_unique = [cut_unique]
    cut_encoder = OrdinalEncoder(categories=cut_unique)
    df_train['cut'] = cut_encoder.fit_transform(df_train[['cut']])
    df_test['cut'] = cut_encoder.fit_transform(df_test[['cut']])
    #Fro Color Varible
    color_unique = df_train['color'].unique()
    color_unique = [color_unique]
    color_encoder = OrdinalEncoder(categories=color_unique)
    df_train['color'] = color_encoder.fit_transform(df_train[['color']])
    df_test['color'] = color_encoder.fit_transform(df_test[['color']])
    #For clarity Varible
    clarity_unique = df_train['clarity'].unique()
    clarity_unique = [clarity_unique]
    clarity_encoder = OrdinalEncoder(categories=clarity_unique)
    df_train['clarity'] = clarity_encoder.fit_transform(df_train[['clarity']])
    df_test['clarity'] = clarity_encoder.fit_transform(df_test[['clarity']])
    # x_data = df_train['x'].values.tolist()
    # print(x_data.count(0))
    # print(df_train['y'].value_counts(0.0))
    # print(df_train['z'].value_counts(0.0))
    # print(df_test['x'].sort_values(ascending=True))#可以观察到长宽高中有0值，为异常值，需要进行替换修改
    # print(df_test['y'].count(0))
    # print(df_test['z'].count(0))
    for column in ['x','y','z']:
        df_train[column] = df_train[column].replace(0,1)
        df_test[column] = df_test[column].replace(0,1)
    # print(df_train.head())
    # print(df_test.head())

    print()
    return df_train,df_test
def product_new_features(args):
    #训练集特征创建
    df_train,df_test = process_data(args)
    df_train['volume'] = df_train['x'] * df_train['y'] * df_train['z']
    df_train['density'] = df_train['carat'] / df_train['volume']
    df_train['table_percentage'] = (df_train['table'] / ((df_train['x'] + df_train['y']) / 2)) * 100
    df_train['depth_percentage'] = (df_train['depth'] / ((df_train['x'] + df_train['y']) / 2)) * 100
    df_train['symmetry'] = (abs(df_train['x'] - df_train['z']) + abs(df_train['y'] - df_train['z'])) / (
                df_train['x'] + df_train['y'] + df_train['z'])
    df_train['surface_area'] = 2 * (
                (df_train['x'] * df_train['y']) + (df_train['x'] * df_train['z']) + (df_train['y'] * df_train['z']))
    df_train['depth_to_table_ratio'] = df_train['depth'] / df_train['table']

    #测试集特征创建
    df_test['volume'] = df_test['x'] * df_test['y'] * df_test['z']
    df_test['density'] = df_test['carat'] / df_test['volume']
    df_test['table_percentage'] = (df_test['table'] / ((df_test['x'] + df_test['y']) / 2)) * 100
    df_test['depth_percentage'] = (df_test['depth'] / ((df_test['x'] + df_test['y']) / 2)) * 100
    df_test['symmetry'] = (abs(df_test['x'] - df_test['z']) + abs(df_test['y'] - df_test['z'])) / (
                df_test['x'] + df_test['y'] + df_test['z'])
    df_test['surface_area'] = 2 * (
                (df_test['x'] * df_test['y']) + (df_test['x'] * df_test['z']) + (df_test['y'] * df_test['z']))
    df_test['depth_to_table_ratio'] = df_test['depth'] / df_test['table']
    # print(df_train.head())
    # print(df_test.head())
    return df_train,df_test
# Split data into x_train and y_train
def split_data(args):
    df_train,df_test = product_new_features(args)
    y_train = df_train['price']
    x_train = df_train.drop('price',axis=1)#按照列进行删除价格这一个特征
    x_test = df_test
    data_columns = x_train.columns
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)

    return x_train,x_val,y_train,y_val,data_columns

    print()

def creating_model_LGBR(args):
    x_train, x_val, y_train, y_val,data_columns = split_data(args)
    lgb_params = {
        'objective':'regression',
        'metric':'mse',
    }
    kf = KFold(n_splits=10,shuffle=True,random_state=0)
    model = lgb.LGBMRegressor(**lgb_params, importance_type='gain')
    # model = xgb.XGBRegressor(**lgb_params)
    cv_results = cross_validate(model,x_train,y_train,scoring='neg_root_mean_squared_error',cv=kf,return_estimator=True)
    print(cv_results.keys())
    feature_importances = [estimator.feature_importances_ for estimator in cv_results["estimator"]]
    feature_importances = np.mean(feature_importances, axis=0)
    feature_importances.shape
    plot = False
    if plot:
        plt.barh(data_columns,feature_importances)#画出来柱状图
        plt.xlabel("Feature importance")
        plt.ylabel("Feature name")
        plt.title("Feature importance using LGBM and cross validation")
        plt.show()
    #按照列来进行求平均
    y_pred = np.mean([model.predict(x_val) for model in cv_results['estimator']], axis=0)
    Y_pred = [model.predict(x_val) for model in cv_results['estimator']]
    print()

def creating_model_XGBR(args):
    x_train, x_val, y_train, y_val, data_columns = split_data(args)
    lgb_params = {
        'objective': 'regression',
        'metric': 'mse',
    }
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    model = lgb.LGBMRegressor(**lgb_params, importance_type='gain')
    model = xgb.XGBRegressor(**lgb_params,importance_type='gain')
    cv_results = cross_validate(model, x_train, y_train, scoring='neg_root_mean_squared_error', cv=kf,
                                return_estimator=True)
    print(cv_results['estimator'])
    feature_importances = [estimator.feature_importances_ for estimator in cv_results["estimator"]]
    feature_importances = np.mean(feature_importances, axis=0)
    feature_importances.shape

    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',type=str,default='E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e8')
    args = parser.parse_args()
    creating_model_LGBR(args)