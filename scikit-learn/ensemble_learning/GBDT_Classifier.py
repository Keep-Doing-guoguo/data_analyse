import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance#特征重要性排列顺序
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    #"loss": "squared_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

test_score = np.zeros((params["n_estimators"],), dtype=np.float64)


for i,y_pred in enumerate(reg.staged_predict(X_test)):#返回每个基分类器的预测数据集X的结果。
    test_score[i] = mean_squared_error(y_test,y_pred)

fig = plt.figure(figsize=(6,6))
plt.subplot(1,1,1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1,reg.train_score_,'b-',label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1,test_score,'r-',label = 'Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.tight_layout()
plt.show()




feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)#返回的是索引列表
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.barh(pos,feature_importance[sorted_idx],align='center')
plt.yticks(pos,np.array(diabetes.feature_names)[sorted_idx])
plt.title('Feature Importance (MDI)')

#n_repeats=10:排列特征的次数。置换特征的次数。个人理解可能是交换计算特征重要性的次数。
result = permutation_importance(reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()#获得到特征影响因素最大的下表索引，
abc = result.importances[sorted_idx].T
plt.subplot(1,2,2)
plt.boxplot(result.importances[sorted_idx].T,vert=False,labels=np.array(diabetes.feature_names)[sorted_idx])
plt.title('Permutation Importance (test set)')
plt.tight_layout()
plt.show()

print()