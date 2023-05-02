# Run the following line once to install. You may need to restart your runtime afterwards:
#######################  First import libraries and data#######################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from IPython import display
# display.set_matplotlib_formats('svg')

feather = pd.read_feather('E:/ZGW/Data/house_sales.ftr')
data = feather
# print(data.shape)
# print(data.head())

####################### We drop columns that at least 30% values are null to simplify our EDA#######################
null_sum = data.isnull().sum()
data.columns[null_sum < len(data) * 0.3]
data.drop(columns = data.columns[null_sum > len(data) * 0.3],inplace=True)
print(data.dtypes)


####################### Convert currency from string format such as $1,000,000 to float.#######################
