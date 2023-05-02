import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns


rng = np.random.RandomState(1)
X = np.linspace(0,6,100).reshape(100,-1)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0,0.1,X.shape[0])

regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300,random_state=rng)
regr_1.fit(X, y)
regr_2.fit(X, y)

y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

colors = sns.color_palette('colorblind')

plt.figure()
plt.scatter(X,y,color=colors[0],label = 'trainning samples')#原始数据显示散点图
plt.plot(X,y_1,color = colors[1], label="n_estimators=300", linewidth=2)
plt.plot(X,y_2,color = colors[2], label="n_estimators=300", linewidth=2)
plt.xlabel('data')
plt.ylabel('target')
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()



print()