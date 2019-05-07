# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)

# plt.rcParams['font.sans-serif']=['simhei'] # 显示中文

# 获得特征矩阵和标签矩阵
def get_Mat(path):
    dataSet = pd.read_csv(path, header = None, sep='\t')
    xMat = np.mat(dataSet.iloc[:,:-1].values)
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    return xMat,yMat

#数据集可视化函数
def showPlot(xMat,yMat):
    x=np.array(xMat[:,0])
    y=np.array(xMat[:,1])
    label = np.array(yMat)
    plt.scatter(x, y, c=label)
    plt.title('Single Decision-Tree Testing Data')
    plt.show()

"""
函数功能：单层决策树分类函数
参数说明:
    xMat: 数据矩阵
    i: 第i列，也就是第几个特征
    Q: 阈值
    S: 标志
返回:
    re: 分类结果
"""
def Classify0(xMat, i, Q, S):
    re = np.ones((xMat.shape[0], 1))  # 初始化re为1
    if S == 'lt':
        re[xMat[:, i] <= Q] = -1  # 如果小于阈值,则赋值为-1
    else:
        re[xMat[:, i] > Q] = -1  # 如果大于阈值,则赋值为-1
    return re

"""
函数功能：找到数据集上最佳的单层决策树
参数说明:
    xMat：特征矩阵
    yMat：标签矩阵
    D：样本权重
返回:
    bestStump：最佳单层决策树信息
    minE：最小误差
    bestClas：最佳的分类结果
"""
def get_Stump(xMat, yMat, D):
    m, n = xMat.shape  # m为样本个数，n为特征数
    Steps = 10  # 初始化一个步数
    bestStump = {}  # 用字典形式来储存树桩信息
    bestClas = np.mat(np.zeros((m, 1)))  # 初始化分类结果为1
    minE = np.inf  # 最小误差初始化为正无穷大
    for i in range(n):  # 遍历所有特征
        Min = xMat[:, i].min()  # 找到特征中最小值
        Max = xMat[:, i].max()  # 找到特征中最大值
        stepSize = (Max - Min) / Steps  # 计算步长
        for j in range(-1, int(Steps)+1):
            for S in ['lt', 'gt']:  # 小于和大于的情况，均遍历。lt:less than，gt:greater than
                Q = (Min + j * stepSize)  # 计算阈值
                # print Q
                re = Classify0(xMat, i, Q, S)  # 计算分类结果
                err = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                err[re == yMat] = 0  # 分类正确的,赋值为0
                eca = D.T * err  # 计算误差
                # print(f'切分特征: {i}, 阈值:{np.round(Q,2)}, 标志:{S}, 权重误差:{np.round(eca,3)}')
                if eca < minE:  # 找到误差最小的分类方式
                    minE = eca
                    bestClas = re.copy()
                    bestStump['特征列'] = i
                    bestStump['阈值'] = Q
                    bestStump['标志'] = S
    return bestStump, minE, bestClas

"""
函数功能：基于单层决策树的AdaBoost训练过程
参数说明：
    xMat：特征矩阵
    yMat：标签矩阵
    maxC：最大迭代次数
返回：
    weakClass：弱分类器信息
    aggClass：类别估计值（其实就是更改了标签的估计值）
"""
def Ada_train(xMat, yMat, maxC = 40):
    weakClass = []
    m = xMat.shape[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化权重
    aggClass = np.mat(np.zeros((m,1)))
    for i in range(maxC):
        Stump, error, bestClas = get_Stump(xMat, yMat,D)  # 构建单层决策树
        #print(f"D:{D.T}")
        alpha=float(0.5 * np.log((1 - error) / max(error, 1e-16)))  # 计算弱分类器权重alpha
        Stump['alpha'] = np.round(alpha,2)  # 存储弱学习算法权重,保留两位小数
        weakClass.append(Stump)  # 存储单层决策树
        #print("bestClas: ", bestClas.T)
        expon = np.multiply(-1 * alpha *yMat, bestClas)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # 根据样本权重公式，更新样本权重
        aggClass += alpha * bestClas  # 更新累计类别估计值
        #print(f"aggClass: {aggClass.T}" )
        aggErr = np.multiply(np.sign(aggClass) != yMat, np.ones((m,1)))  # 计算误差
        errRate = aggErr.sum() / m
        #print(f"分类错误率: {errRate}")
        if errRate == 0:
            break  # 误差为0，退出循环
    return weakClass, aggClass

"""
函数功能：AdaBoost分类函数
参数说明：
    data: 待分类样例
    classifys: 训练好的分类器
返回：
    分类结果
"""
def AdaClassify(data, classifys):
    dataMat = np.mat(data)
    m = dataMat.shape[0]
    aggClass = np.mat(np.zeros((m,1)))
    for i in range(len(classifys)): #遍历所有分类器，进行分类
        classEst = Classify0(dataMat,
                             classifys[i]['特征列'],
                             classifys[i]['阈值'],
                             classifys[i]['标志'])
        aggClass += classifys[i]['alpha'] * classEst
        #print(aggClass)
    return np.sign(aggClass)

"""
函数功能：计算训练测试准确率
"""
def calAcc(maxC=40):
    train_xMat, train_yMat = get_Mat('horseColicTraining2.txt')
    m = train_xMat.shape[0]
    weakClass, aggClass = Ada_train(train_xMat, train_yMat, maxC)
    yhat = AdaClassify(train_xMat, weakClass)
    train_re = 0
    for i in range(m):
        if yhat[i] == train_yMat[i]:
            train_re += 1
    train_acc = train_re / m
    print '训练集准确率为', train_acc

    test_re = 0
    test_xMat, test_yMat = get_Mat('horseColicTest2.txt')
    n = test_xMat.shape[0]
    yhat = AdaClassify(test_xMat, weakClass)
    for i in range(n):
        if yhat[i] == test_yMat[i]:
            test_re += 1
    test_acc = test_re / n
    print '测试集准确率为', test_acc
    return train_acc, test_acc

# -------------------------------------------------------------------
# 测试数据
# xMat,yMat = get_Mat('simpdata.txt')
#
# print xMat, '\n', yMat
#
# # showPlot(xMat, yMat)
#
# # m = xMat.shape[0]
# # D = np.mat(np.ones((m, 1)) / m)  # 初始化样本权重（每个样本权重相等）
# # bestStump, minE, bestClass = get_Stump(xMat, yMat, D)
# #
# # # print json.dumps(bestStump, encoding="UTF-8", ensure_ascii=False)
# # print bestClass
#
# weakClass, aggClass = Ada_train(xMat, yMat, maxC = 40)
#
# # print json.dumps(weakClass, encoding="UTF-8", ensure_ascii=False)
# # print '\n', aggClass
#
# flag = AdaClassify([1,2], weakClass)
# print flag
# -------------------------------------------------------------------
# 病马数据集测试
# train = pd.read_csv('horseColicTraining2.txt', header=None, sep='\t')
# test = pd.read_csv('horseColicTest2.txt', header=None, sep='\t')
#
# print train.head()
#
# print '\n', train.iloc[:, -1].value_counts()
#
# print train.shape

calAcc(maxC = 40)

# Cycles=[1,10,50,100,500,1000,10000]
# train_acc=[]
# test_acc=[]
# for maxC in Cycles:
#     a, b=calAcc(maxC)
#     train_acc.append(round(a * 100, 2))
#     test_acc.append(round(b * 100, 2))
# df=pd.DataFrame({'分类器数目':Cycles,'训练集准确率':train_acc,'测试集准确率':test_acc})
# print df