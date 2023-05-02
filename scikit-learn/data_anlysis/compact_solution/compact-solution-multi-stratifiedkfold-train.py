#对于不知道的人来说，这种竞争特征是极其嘈杂的数据和异常值。您可以在讨论线程中找到很多有趣的主题
#由于上述原因，公共排行榜具有误导性，许多没有交叉验证的简单方法幸运地处于领先地位，
#并且还偏向于其他竞争对手遵循他们的方法。
#我不得不承认，在比赛开始时，我是他们中的一员，因为我认为我走错了方向。
#删除异常值 - 对每列运行 IQR 并删除具有异常值的行（仅 20 条记录）。
#异常值去除对于最终模型性能并不那么重要，但对于CV很重要，因为RMSE指标对异常值非常敏感，可能导致误导性结果。
#分段模型（我喜欢 @PRASAD 中的术语）- 根据不同的时期拆分模型（基于制作列）。最佳分数是将数据分成 4 个周期。
#多层KFold - 使用不同的种子多次运行分层KFold。
# 这让我有信心获得有关模型性能的良好统计数据（平均分数 +/- SD）。第二个关键点是“统计”简历的选择。 由于“分段模型”方法，我不得不根据“制造”列对数据进行分层，以获得一致的结果。

import re

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)#警告过滤器
pd.set_option('plotting.backend', 'plotly')# 更改后端绘图方式
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)#设置最大列数
pd.set_option('display.width', 1000)#设置最大宽度
pd.set_option('display.expand_frame_repr', False)#设置不转换行
pd.set_option('display.max_colwidth', -1)#最大列字符数

#Notebook Config
RUN_EVALUATION   = True  # Runs or not the evaluation scripts
CREAT_SUBMISSION = True   # Creates or not the submission.csv file
#E:\ZGW\PycharmProjects1\pythonProject1\Data_an\compact_solution\test.csv
#Load Data
original_df = pd.read_csv('E:/ZGW/PycharmProjects1/pythonProject1/Data_an/compact_solution/train.csv')
train_df = pd.read_csv('E:/ZGW/PycharmProjects1/pythonProject1/Data_an/compact_solution/train.csv')
test_df = pd.read_csv('E:/ZGW/PycharmProjects1/pythonProject1/Data_an/compact_solution/test.csv')

#Remove made outliers
train_df = train_df[train_df['made'] <= 2021]

#Preprocessing
def model_predict(train_df,original_df,test_df,verbose=False):
    train_df = train_df.copy()
    original_df = original_df.copy()
    test_df = test_df.copy()
#https://www.kaggle.com/competitions/playground-series-s3e6/data?select=train.csv
    # -----------------------
    # CLEANING
    # -----------------------
    outlier_ids = [15334, 5659, 2113, 3608, 19124, 19748, 21400, 2107,
                   3995, 15068, 18926, 3828, 4909, 12858, 13633, 13642,
                   17168, 19994, 14878, 17629]
    train_df[~train_df['id'].isin(outlier_ids)]

    # -----------------------
    # PRE-PROCESSING
    # -----------------------
    train_df = train_df.drop(columns=['id'])#删除列名为id的这一项
    train_df['original'] = 0
    original_df['original'] = 1
    test_df['original'] = 0

    train_df = pd.concat([train_df, original_df])
    # Create yearly stats
    made = train_df.groupby(['made', 'original'], as_index=False) \
        .agg(**{'count': pd.NamedAgg(column='made', aggfunc='count')})

    # Merge with 'made' dataframe
    train_df = pd.merge(train_df, made, on=['made', 'original'], how='left')
    test_df = pd.merge(test_df, made, on=['made', 'original'], how='left')

    # -----------------------
    # ENSEMBLE MODEL
    # -----------------------
    rf = RandomForestRegressor(random_state=0)
    gb = GradientBoostingRegressor(random_state=0)
    xgb = XGBRegressor(n_estimators=300,
                       gamma=0.1,
                       random_state=0)
    cb = CatBoostRegressor(l2_leaf_reg=1,
                           depth=6,
                           verbose=False,
                           random_state=0)
    estimators = [('rf', rf),
                  ('gb', gb),
                  ('xgb', xgb),
                  ('cb', cb)
                  ]
    model = VotingRegressor(estimators=estimators)

    # -----------------------
    # RANGES SPLIT
    # -----------------------
    ranges = [
        lambda df: df['made'] <= 2005,
        lambda df: (df['made'] > 2005) & (df['made'] <= 2007),
        lambda df: (df['made'] > 2007) & (df['made'] <= 2015),
        lambda df: df['made'] > 2015
    ]

    # -----------------------
    # PREDICTION
    # -----------------------
    for i, range_ in enumerate(ranges):
        train_ = train_df[range_(train_df)]
        test_ = test_df[range_(test_df)]
        model.fit(train_.drop(columns=['price']), train_['price'])
        model_scores = []
        # GET ESTIMATORS SCORES
        if verbose:
            print(f' - Range: {i + 1}/{len(ranges)}')
            for estimator in model.estimators_:
                score = mean_squared_error(estimator.predict(test_[train_.drop(columns=['price']).columns]),
                                           test_['price'], squared=False)
                model_scores.append((score))
                best_estimator = estimators[np.argmin(model_scores)][0]
            print(f'    - Estimators Score: {model_scores} | BEST: {best_estimator}')
        pred = model.predict(test_[train_.drop(columns=['price']).columns])
        test_df.loc[range_(test_df), 'price'] = pred.clip(0)
    return test_df['price']

#Model Evaluation
#PART-1: Train-Test Split (80/20) [Fast Evaluation]
if RUN_EVALUATION:
    # NOTE:  Remove 2002 because has only 1 record and it creates problem with the CV
    #ttt = train_df['made'].isin([2002])
    train_df_ = train_df[~train_df['made'].isin([2002])]#取反之后就将不包含2002的数据给取出来了。

    X_train, X_val = train_test_split(train_df_,
                                      test_size=0.20,
                                      shuffle=True,
                                      random_state=0,
                                      stratify=train_df_['made']
    )
    prediction = model_predict(X_train, original_df, X_val)
    score = mean_squared_error(prediction, X_val["price"], squared=False)
    print('======= TRAIN-TEST SPLIT SCORE ==========')
    print(f'* RMSE:{score}')
#PART-2: Multi CrossValidation [Slow Evaluation]
if RUN_EVALUATION:
    # Cross Validation Config
    N_SPLITS = 5
    N_RUNS = 3

    # NOTE:  Remove 2002 because has only 1 record and it creates problem with the CV
    train_df_ = train_df[~train_df['made'].isin([2002])]

    cv_results = []
    for run_id in range(1, N_RUNS+1):
        print('-----------------------------')
        print(f'* Run ID: {run_id}/{N_RUNS}')
        print('-----------------------------')
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=run_id)
        for i, (train_index, test_index) in enumerate(skf.split(train_df_, train_df_['made'])):
            X_train = train_df_.iloc[train_index]
            X_val = train_df_.iloc[test_index]

            test_prediction = model_predict(X_train, original_df, X_val)
            test_score = mean_squared_error(test_prediction, X_val["price"], squared=False)

            cv_results.append(test_score)
            mean_cv_scores = np.mean(cv_results)
            std_cv_scores = np.std(cv_results)

            print(f' * Iteration {i+1}/{N_SPLITS} | TEST SCORE: {test_score}')


    print('=========================')
    print('RESULTS')
    print('=========================')
    print("Cross-validation results ({}-fold):".format(N_SPLITS))
    print("Average Score = {:.2f} +/- {:.2f}".format(mean_cv_scores, std_cv_scores))
print()