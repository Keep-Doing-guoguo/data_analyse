import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import gc
import re as re
from collections import Counter

from tqdm.auto import tqdm
import math
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import warnings
warnings.filterwarnings('ignore')

import time
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

tqdm.pandas()

rc = {
    "axes.facecolor": "#FFF9ED",
    "figure.facecolor": "#FFF9ED",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc=rc)

from colorama import Style, Fore
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
mgt = Style.BRIGHT + Fore.MAGENTA
gld = Style.BRIGHT + Fore.YELLOW
res = Style.RESET_ALL

pd.set_option('display.max_rows', 100)
#E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e13/train.csv
train = pd.read_csv('E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e13/train.csv')
test = pd.read_csv('E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e13/test.csv')
original = pd.read_csv('E:/ZGW/PycharmProjects1/pythonProject1/Data_an/s3e13/trainn.csv')


# summary table function
def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values
    summ['%missing'] = df.isnull().sum().values / len(df) * 100
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values

    return summ

print(test.info())
summary(train)

# select X variables
all_cols = test.select_dtypes(include=['float64','int64']).columns.tolist()
all_cols.remove('id')


plt.figure(figsize=(14,10))
for idx,column in enumerate(all_cols[:2]):
    plt.subplot(2,1,idx+1)
    sns.countplot(x=column, hue="prognosis", data=train, palette="YlOrRd")
    plt.title(f"{column} Distribution")
    plt.legend(loc = 'upper right', bbox_to_anchor = (1, 1))
    plt.tight_layout()


print()