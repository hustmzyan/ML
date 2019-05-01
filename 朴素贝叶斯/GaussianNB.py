# -*- coding: utf-8 -*-
"""
鸢尾花特征分类实例分类
"""

import numpy as np
import pandas as pd
import random

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)


# 对数据集进行随机切分得到训练集、测试集
def randSplit(dataSet, rate):
    l = list(dataSet.index) #提取出索引
    random.shuffle(l) #随机打乱索引
    dataSet.index = l #将打乱后的索引重新赋值给原数据集
    n = dataSet.shape[0] #总行数
    m = int(n * rate) #训练集的数量
    train = dataSet.loc[range(m), :] #提取前m个记录作为训练集
    test = dataSet.loc[range(m, n), :] #剩下的作为测试集
    dataSet.index = range(dataSet.shape[0]) #更新原数据集的索引
    test.index = range(test.shape[0]) #更新测试集的索引
    return train, test

def gnb_classify(train,test):
    labels = train.iloc[:,-1].value_counts().index #提取训练集的标签种类
    mean =[] #存放每个类别的均值
    std =[] #存放每个类别的方差
    result = [] #存放测试集的预测结果
    for i in labels:
        item = train.loc[train.iloc[:,-1]==i,:] #分别提取出每一种类别
        m = item.iloc[:,:-1].mean() #当前类别的平均值
        s = np.sum((item.iloc[:,:-1]-m)**2)/(item.shape[0]) #当前类别的方差
        mean.append(m) #将当前类别的平均值追加至列表
        std.append(s) #将当前类别的方差追加至列表
    means = pd.DataFrame(mean,index=labels) #变成DF格式，索引为类标签
    stds = pd.DataFrame(std,index=labels) #变成DF格式，索引为类标签
    for j in range(test.shape[0]):
        iset = test.iloc[j,:-1].tolist() #当前测试实例
        iprob = np.exp(-1*(iset-means)**2/(stds*2))/(np.sqrt(2*np.pi*stds)) #正态分布公式
        prob = 1 #初始化当前实例总概率
        for k in range(test.shape[1]-1): #遍历每个特征
            prob *= iprob[k] #特征概率之积即为当前实例概率
            cla = prob.index[np.argmax(prob.values)] #返回最大概率的类别
        result.append(cla)
    test['predict']=result
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean() #计算预测准确率
    return test, acc

# 导入数据集
dataSet = pd.read_csv('iris.txt', header = None)
# print dataSet

train, test = randSplit(dataSet, 0.8)
print train
print '---\n'
print test

test, acc= gnb_classify(train, test)
# print test
print "模型预测准确率为", acc

