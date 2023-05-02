
# 多分类问题

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier#神经网络分类器
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier#随机森林，adaboost
from sklearn.neighbors import KNeighborsClassifier#k近邻
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
import warnings
from sklearn.tree import DecisionTreeClassifier#决策树分类器

plt.rc('font', family='Times New Roman')
warnings.filterwarnings("ignore")

paTH = '/Data_an/industry/train data.xlsx'
df = pd.read_excel(paTH)
data_train = pd.DataFrame(df,columns=['机器编号','机器质量等级','室温（K）','机器温度（K）','转速（rpm）',
								  '扭矩（Nm）','使用时长（min）','是否发生故障','具体故障类别'])
#在这里一共取出来了4列数据信息。
le = data_train['机器质量等级']
T = data_train['室温（K）']
T_machine = data_train['机器温度（K）']
error_type = data_train['具体故障类别']


#查看数据的信息


M_sum = data_train.shape[0]#查看总共的数据

# 建立温升,插入数据中，定名 Tr。两列数据进行相减。
sub_t = T_machine - T
data_train.insert(loc=9,column='Tr',value=sub_t)#在数据中第九列中插入一列数据。

# 对机器质量进行处理，分为三列 L M H
#在这里使用的是人工处理方法，one-hot编码方式。
q_ls = []
for i in range(M_sum):
	if le[i] == 'L':
		q_ls.append([1,0,0])
		continue
	elif le[i] == 'M':
		q_ls.append([0,1,0])
		continue
	elif le[i] == 'H':
		q_ls.append([0,0,1])
print()
q_ls = np.array(q_ls)

train_set = np.array([data_train['转速（rpm）'],
				      data_train['扭矩（Nm）'],
				      data_train['使用时长（min）'],
					  data_train['Tr']])
train_set = train_set.T
# 归一化
scale = MinMaxScaler()
print("未归一化特征数据集及维度：")
print(train_set,train_set.shape,sep='\n')
train_set = scale.fit_transform(train_set)
print("归一化特征数据集：")
print(train_set,train_set.shape,sep='\n')
train_set = np.hstack((q_ls,train_set))#按照水平方向上来进行拼接。

Unique = error_type.unique()

# 对故障数据进行处理。这里是对数据进行处理。
error_type_ls = []
for i in range(M_sum):
	if error_type[i] == 'TWF':
		error_type_ls.append([1])
	elif error_type[i] == 'HDF':
		error_type_ls.append([2])
	elif error_type[i] == 'PWF':
		error_type_ls.append([3])
	elif error_type[i] == 'OSF':
		error_type_ls.append([4])
	elif error_type[i] == 'RNF':
		error_type_ls.append([5])
	else:
		error_type_ls.append([0])

label_set = np.array(error_type_ls)     # 标签数据集

from sklearn.model_selection import train_test_split as TTS#数据集进行划分




# 数据集 8:1:1
split1 = int(M_sum*0.8)
split2 = int(M_sum*0.9)
train_x = train_set[:split1,:]
train_y = label_set[:split1,:]
cv_x = train_set[split1:split2,:]
cv_y = label_set[split1:split2,:]
test_x = train_set[split2:,:]
test_y = label_set[split2:,:]






# 分类器
clf_RF = RandomForestClassifier(n_estimators=120,random_state=40)#随机森林，在这里设置了120颗决策树。
clf_NN = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(15,), random_state=4)#使用神经网络来进行预测
clf_ada = AdaBoostClassifier(n_estimators=100)#
clf_knn = KNeighborsClassifier(n_neighbors=5)#使用k近邻来进行预测
clf_tree = DecisionTreeClassifier()#使用决策树来进行预测

# 模型选择
# 用在验证级上的表现作为衡量标准
classifer_ls = [clf_RF,clf_NN,clf_ada,clf_knn,clf_tree]
for clf in classifer_ls:
	clf.fit(train_x, train_y)
	pred = clf.predict(cv_x)
	C = confusion_matrix(cv_y, pred)
	F1 = f1_score(cv_y, pred,average="micro")
	print(f'clf:{clf}，其中的F1_SCORE为：{F1}')

# 选择MLP作为多元分类器 在测试集上的表现
pred = clf_NN.predict(test_x)
print(test_x.shape)
C = confusion_matrix(test_y, pred)
F1 = f1_score(test_y, pred,average="micro")
plot_confusion_matrix(clf_NN, test_x, test_y, cmap=plt.cm.Blues)
print(f'\n选择MLP作为多元分类器 在测试集上的表现,其F1表现为：{F1}')
plt.show()


pred = clf_NN.predict(train_x)
print(train_x.shape)
C = confusion_matrix(train_y, pred)
F1 = f1_score(train_y, pred,average="micro")
plot_confusion_matrix(clf_NN, train_x, train_y, cmap=plt.cm.Blues)
print(f'\n选择MLP作为多元分类器 在测试集上的表现,其F1表现为：{F1}')
plt.show()