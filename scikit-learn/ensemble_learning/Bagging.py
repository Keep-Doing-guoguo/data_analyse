from numpy import *
import matplotlib.pyplot as plt
import random
from sklearn import tree

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')#line.strip()首先清除掉一些空格，然后按照'\t'进行划分
        for i in range(numFeat - 1):#添加数据
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))#添加数据对应标签
    return dataMat,labelMat

#自助法采样
def rand_train(dataMat,labelMat):
    len_train = len(labelMat)
    train_data = []
    train_label = []
    #抽取样本的次数为样本的数目
    for i in range(len_train):
        index = random.randint(0,len_train-1)
        train_data.append(dataMat[index])
        train_label.append(labelMat[index])
    return train_data,train_label

#决策树学习
#默认并行生成十个基学习器
def bagging_by_tree(dataMat,labelMat,t=10):
    test_data,test_label = loadDataSet('E:/ZGW/PycharmProjects1/pythonProject1/scikit-learn/ensemble_learning/HorseColicData/horseColicTest.txt')
    predict_list = []
    for i in range(t):
        train_data,train_label = rand_train(dataMat,labelMat)
        clf = tree.DecisionTreeClassifier()#初始化决策树模型
        clf.fit(train_data,train_label)#训练模型
        total = []
        y_predicted = clf.predict(test_data)#预测数据
        total.append(y_predicted)
        predict_list.append(total)#结果添加到预测列表当中
    return predict_list,test_label

#计算错误率
def calc_error(predict_list,test_label):
    m,n,k = shape(predict_list)
    #分类问题就使用投票数，投票数占比最多的一个类别。
    predict_label = sum(predict_list,axis=0)
    predict_label = sign(predict_label)#取数字符号（数字前的正负号）.如果为负号就说明类别数为-1，如果为正号就说明类别数为+1.
    for i in range(len(predict_label[0])):
        if predict_label[0][i] == 0:
            tip = random.randint(0,1)
            if tip == 0:
                predict_label[0][i] = 1
            else:
                predict_label[0][i] = -1
    error_count = 0
    for i in range(k):
        if predict_label[0][i] != test_label[i]:
            error_count += 1
    error_rate = error_count / k
    return error_rate
def bagging_by_Onetree(dataMat,labelMat,t=10):
    test_data,test_label = loadDataSet('E:/ZGW/PycharmProjects1/pythonProject1/scikit-learn/ensemble_learning/HorseColicData/horseColicTest.txt')
    train_data,train_label = rand_train(dataMat,labelMat)
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data,train_label)
    y_predicted = clf.predict(test_data)
    error_count = 0
    for i in range(67):
        if y_predicted[i] != test_label[i]:
            error_count += 1
    return error_count/67
if __name__ == "__main__":
    fileName = 'E:/ZGW/PycharmProjects1/pythonProject1/scikit-learn/ensemble_learning/HorseColicData/horseColicTraining.txt'
    dataMat,labelMat =  loadDataSet(fileName)
    train_data,train_label = rand_train(dataMat,labelMat)
    predict_list , test_label = bagging_by_tree(dataMat,labelMat)
    print('单一错误率:',bagging_by_Onetree(dataMat,labelMat))
    print("Bagging错误率：",calc_error(predict_list,test_label))