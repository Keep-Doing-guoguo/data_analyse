#作物产量统计EDA
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px#这一个包是将图像显示到web上面。

# Read Dataset
#E:/ZGW/PycharmProjects1/pythonProject1/Data_an/APY.csv
data = pd.read_csv('E:/ZGW/PycharmProjects1/pythonProject1/Data_an/APY.csv')
# print(data.head())
print(data.shape)
# print(data.info())
'''
Crop        345327 non-null  object 
Production  340388 non-null  float64
这两行是有缺失值的，可以从info里面看出来。
'''
#DATA CLEANING数据清洗
#print(data.isnull().sum())
#直接删除空缺值
new_data = data.dropna()
# print(new_data.shape)
# print(new_data.isnull().sum())
#Number of  crops types in the dataset
print('number of crops: ',new_data['Crop'].nunique())#nunique输出的是长度
print('\n--------------------------------------xxxxxxxxxxxxx-------------------------------\n')
print('crop type: ',new_data['Crop'].unique())#输出里面的类别
#Number of  season in the dataset
print('number of season: ',new_data['Season'].nunique())
print('\n--------------------------------------xxxxxxxxxxxxx-------------------------------\n')
print('crop type: ',new_data['Season'].unique())
print('\n--------------------------------------xxxxxxxxxxxxx-------------------------------\n')
print('State type: ',new_data['State'].unique())#输出里面的类别
#Top 5 States Production in India
print("Top 5 Season in production :\n")
top_state_nosum = data.groupby('Season')['Production'].sum()
top_state = data.groupby('Season')['Production'].sum().nlargest()
print(top_state)
print("Top 5 states in the annul yielding :\n")#年产量排名前 5 的州
top_state = data.groupby('State')['Yield'].sum().nlargest()
print(top_state)

#MAXIMUM YIELDS IN INDIA BASED ON STATES
print("Top 5 Areas in the State :\n")
#top_area = new_data.groupby('State').agg({"Yield":"max"})
#print(new_data['State'].nunique())
top_area = new_data.groupby('State').agg({'Yield':'max'})#这里的意思是取出来的Yield的最大值，基本上是全部取出来了。
print(top_area)

#Visualization for Kerala (Top in Production Rate)
#只要某一个地区的信息
kerala = new_data[new_data['State'] == 'Kerala']
#print(kerala.head())
# Number of crop production in Kerala
# print("Number of crop production : ",kerala["Crop"].nunique())

# fig = px.histogram(kerala,x='Crop',y='Production',color='Season')#画的是直方图
# #fig.update_layout(barmode='group')
#
# fig.show()
#
# fig = px.histogram(kerala, x="Crop", y="Yield", color="Season")
# fig.show()
#
# fig = px.histogram(kerala, x="Crop_Year", y="Production", color="Crop")
# fig.update_layout(barmode='group')
# fig.show()

#Visualization for TamilNadu (Top Second in Production Rate and Top in Yeild rate)
#泰米尔纳德邦的可视化（生产率排名第二，耶尔德率最高）

#只要某一个地区的数据信息
tamil_nadu = new_data[new_data["State"]=='Tamil Nadu']
print(tamil_nadu.info())
# tamil_yeild = px.histogram(tamil_nadu,x='Crop',y='Yield',color='Season')
# tamil_yeild.show()
# tamil_yeild = px.histogram(tamil_nadu, x="Crop", y="Production", color="Season")
# tamil_yeild.show()
tn_top_produciton = tamil_nadu.groupby('Crop')['Production'].sum().nlargest()
#print(tn_top_produciton)
tn_top_produciton = tamil_nadu.groupby('Crop')['Production'].sum().sort_values(ascending=False)
print(tn_top_produciton)
tn_top_production = tamil_nadu.groupby(['Season'])['Yield'].sum().sort_values(ascending=False)
print(tn_top_production)
tamil_yeild = px.histogram(tamil_nadu, x="Season", y="Yield")
tamil_yeild.show()