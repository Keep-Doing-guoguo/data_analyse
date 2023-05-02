import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import shap
import eli5
pd.set_option('display.max_rows', 200)
pd.set_option('expand_frame_repr', False)
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import make_scorer,roc_auc_score
from sklearn.preprocessing import MinMaxScaler


train_path = 'E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e7/train.csv'
test_path = 'E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e7/test.csv'
orginal_path = 'E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e7/train__dataset.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
org_df = pd.read_csv(orginal_path)
# print(train.head())
# print(train.shape)
# print(test.shape)
print(train.info())
print(test.info())
# print(train.isnull().sum())
# print(train.isnull().any())
# print(train.describe().T)
# print(train.columns)
# print(test.columns)
#print(test['booking_status'].unique())
#这是一个分类问题，将会考虑训练集和测试集中数据分布的情况。所有使用strightflod.并且还是一个二分类型的问题。
train = train.drop('id',axis=1)
df = pd.concat([train,org_df],axis=0)
df.reset_index(drop=True,inplace=True)
df.drop_duplicates(inplace=True)
def feature_engineering(df):
    df['family_size'] = df['no_of_adults'] + df['no_of_children']
    df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    df['total_bookings'] = df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled']
    df['total_cost'] = df['total_nights'] * df['avg_price_per_room']

    return df
'''
其实这样子做的还有点麻烦，可以直接删除掉data大于28的数据，或者将data大于28的数据全部替换为28.
'''
def correct_days_in_month(df):
    df['arrival_year_month'] = pd.to_datetime(df['arrival_year'].astype(str)+df['arrival_month'].astype(str),format='%Y%m')
    df_arrival_year_month_type = df.loc[:,'arrival_year_month'].dt.days_in_month
    df.loc[df.arrival_date > df.arrival_year_month.dt.days_in_month, 'arrival_date'] = df.arrival_year_month.dt.days_in_month
    df.drop(columns='arrival_year_month',inplace=True)
    return df

'''
删除一些有辨识度的异常值
'''
def removing_odd_data(df):
    df = df[(df['total_nights'] > 0) & (df['family_size'] > 0)]
    return df


df = feature_engineering(df)
df = correct_days_in_month(df)
df = removing_odd_data(df)
X = df.drop(['booking_status'],axis=1)
y = df['booking_status']
#保持测试集与整个数据集里result的数据分类比例一致。stratify=y
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,stratify=y)
print(f'test target distribution => {y_test.value_counts()/y_test.shape[0]}\ntrain target distribution => {y_train.value_counts()/y_train.shape[0]}')
def make_mi_scores(X,y,discrete_features):
    mi_scores = mutual_info_classif(X,y,discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores,name='Mi_Scores',index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)#上升=False，
    return mi_scores
#创建离散变量
discrete_features = np.array([len(np.unique(X[col])) < 10 for col in X.columns])
mi_scores = make_mi_scores(X, y, discrete_features = discrete_features)
print()
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)#上升
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width,scores)
    plt.yticks(width,ticks)
    plt.title("Mutual Information Scores")
    plt.show()
plt.figure(dpi=100, figsize=(8, 5))
#plot_mi_scores(mi_scores)
mi_df = pd.DataFrame(mi_scores).rename(columns={'index':'features','Mi_Scores':'mi_scores'})
# print(mi_df)
mi_df['mi_scores'] = MinMaxScaler().fit_transform(mi_df[['mi_scores']])
# print(mi_df)





#print(mi_scores)
print()
print()