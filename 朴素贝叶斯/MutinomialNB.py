# -*- coding: utf-8 -*-
"""
文档分类实例分类
"""
from functools import reduce
import numpy as np
import pandas as pd
import random

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)

def loadDataSet():
    dataSet=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
             ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
             ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
             ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
             ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
             ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']] #切分好的词条
    classVec = [0,1,0,1,0,1] #类别标签向量，1代表侮辱性词汇，0代表非侮辱性词汇
    return dataSet,classVec

def createVocabList(dataSet):
    vocabSet = set() #创建一个空的集合
    for doc in dataSet: #遍历dataSet中的每一条言论
        vocabSet = vocabSet | set(doc) #取并集
        vocabList = list(vocabSet)
    return vocabList

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList) #创建一个其中所含元素都为0的向量
    for word in inputSet: #遍历每个词条
        if word in vocabList: #如果词条存在于词汇表中，则变为1
            returnVec[vocabList.index(word)] = 1
        else:
            print word, " is not in my Vocabulary!"
    return returnVec #返回文档向量

def get_trainMat(dataSet):
    trainMat = [] #初始化向量列表
    vocabList = createVocabList(dataSet) #生成词汇表
    for inputSet in dataSet: #遍历样本词条中的每一条样本
        returnVec=setOfWords2Vec(vocabList, inputSet) #将当前词条向量化
        trainMat.append(returnVec) #追加到向量列表中
    return trainMat

# def trainNB(trainMat,classVec):
#     n = len(trainMat) #计算训练的文档数目
#     m = len(trainMat[0]) #计算每篇文档的词条数
#     pAb = sum(classVec)/n #文档属于侮辱类的概率
#     p0Num = np.zeros(m) #词条出现数初始化为0
#     p1Num = np.zeros(m) #词条出现数初始化为0
#     p0Denom = 0 #分母初始化为0
#     p1Denom = 0 #分母初始化为0
#     for i in range(n): #遍历每一个文档
#         if classVec[i] == 1: #统计属于侮辱类的条件概率所需的数据
#             p1Num += trainMat[i]
#             p1Denom += sum(trainMat[i])
#         else: #统计属于非侮辱类的条件概率所需的数据
#             p0Num += trainMat[i]
#             p0Denom += sum(trainMat[i])
#     p1V = p1Num/p1Denom
#     p0V = p0Num/p0Denom
#     return p0V,p1V,pAb #返回属于非侮辱类,侮辱类和文档属于侮辱类的概率

# 添加拉普拉斯平滑
def trainNB_update(trainMat,classVec):
    n = len(trainMat) #计算训练的文档数目
    m = len(trainMat[0]) #计算每篇文档的词条数
    pAb = sum(classVec)/n #文档属于侮辱类的概率
    p0Num = np.ones(m) #词条出现数初始化为1
    p1Num = np.ones(m) #词条出现数初始化为1
    p0Denom = 2 #分母初始化为2
    p1Denom = 2 #分母初始化为2
    for i in range(n): #遍历每一个文档
        if classVec[i] == 1: #统计属于侮辱类的条件概率所需的数据
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else: #统计属于非侮辱类的条件概率所需的数据
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1V = np.log(p1Num/p1Denom)
    p0V = np.log(p0Num/p0Denom)
    return p0V,p1V,pAb #返回属于非侮辱类,侮辱类和文档属于侮辱类的概率

def classifyNB(vec2Classify, p0V, p1V, pAb):
    p1 = sum(vec2Classify * p1V) + np.log(pAb)     #对应元素相乘
    p0 = sum(vec2Classify * p0V) + np.log(1 - pAb) #对应元素相乘
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB(testVec):
    dataSet,classVec = loadDataSet() #创建实验样本
    vocabList = createVocabList(dataSet) #创建词汇表
    trainMat= get_trainMat(dataSet) #将实验样本向量化
    p0V,p1V,pAb = trainNB_update(trainMat,classVec) #训练朴素贝叶斯分类器
    thisone = setOfWords2Vec(vocabList, testVec) #测试样本向量化
    if classifyNB(thisone,p0V,p1V,pAb):
        print testVec,'属于侮辱类' #执行分类并打印分类结果
    else:
        print testVec,'属于非侮辱类'  #执行分类并打印分类结果


if __name__ == '__main__':
    # 测试样本1
    testVec1 = ['love', 'my', 'dalmation']
    testingNB(testVec1)

    # 测试样本2
    testVec2 = ['stupid', 'garbage']
    testingNB(testVec2)