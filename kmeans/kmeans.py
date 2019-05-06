# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)


"""
函数功能：计算两个数据集之间的欧氏距离
输入：两个array数据集
返回：两个数据集之间的欧式距离（此处用距离的平方代替）
"""
def distEclud(arrA, arrB):
    d = arrA - arrB
    dist = np.sum(np.power(d, 2), axis=1)
    return dist

"""
函数功能：随机生成k个质心
参数说明：
    dataSet：包含标签的数据集
    k：簇的个数
返回：
    data_cent：k个质心
"""
def randCent(dataSet, k):
    n = dataSet.shape[1]
    data_min = dataSet.iloc[:, :n-1].min()
    data_max = dataSet.iloc[:, :n-1].max()
    # print data_min
    # print data_max
    data_cent = np.random.uniform(data_min, data_max, (k, n-1))
    return data_cent

"""
函数功能：k-均值聚类算法
参数说明：
    dataSet：带标签数据集
    k：簇的个数
    distMeas：距离计算函数
    createCent: 随机质心生成函数
返回：
    centroids: 质心
    result_set: 所有数据划分结果
"""
def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    m, n = dataSet.shape
    # 随机生成质心
    centroids = createCent(dataSet, k)
    # clusterAssment
    #   col1:用于存放各点到质心的距离，
    #   col2:用于存放最近一次计算的根据最短距离求得的质心索引
    #   col3:用于存放上一次计算的根据最短距离求得的质心索引，做对比
    clusterAssment = np.zeros((m, 3))
    clusterAssment[:, 0] = np.inf # 距离初始化为最大值
    clusterAssment[:, 1: 3] = -1 # 索引全部初始化为-1
    # 将clusterAssment 与 dataSet 连接起来
    result_set = pd.concat([dataSet, pd.DataFrame(clusterAssment)], axis=1, ignore_index = True)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            dist = distMeas(dataSet.iloc[i, :n-1].values, centroids) # 求与质心的距离
            result_set.iloc[i, n] = dist.min() # 取最短距离
            result_set.iloc[i, n+1] = np.where(dist == dist.min())[0] # 取索引
        clusterChanged = not (result_set.iloc[:, -1] == result_set.iloc[:, -2]).all() # all()只返回单个值
        # 更新数据
        if clusterChanged:
            # 重新计算质心
            cent_df = result_set.groupby(n+1).mean()
            centroids = cent_df.iloc[:, :n-1].values
            result_set.iloc[:, -1] = result_set.iloc[:, -2] # 更新质心索引
    return centroids, result_set

def kMeans_l(dataSet, k, distMeas = distEclud, createCent = randCent):
    m, n = dataSet.shape
    # 随机生成质心
    centroids = createCent(dataSet, k)
    # clusterAssment
    #   col1:用于存放各点到质心的距离，
    #   col2:用于存放最近一次计算的根据最短距离求得的质心索引
    #   col3:用于存放上一次计算的根据最短距离求得的质心索引，做对比
    clusterAssment = np.zeros((m, 3))
    clusterAssment[:, 0] = np.inf # 距离初始化为最大值
    clusterAssment[:, 1: 3] = -1 # 索引全部初始化为-1
    # 将clusterAssment 与 dataSet 连接起来
    result_set = pd.concat([dataSet, pd.DataFrame(clusterAssment)], axis=1, ignore_index = True)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            dist = distMeas(dataSet.iloc[i, :n-1].values, centroids) # 求与质心的距离
            result_set.iloc[i, n] = dist.min() # 取最短距离
            result_set.iloc[i, n+1] = np.where(dist == dist.min())[0] # 取索引
        clusterChanged = not (result_set.iloc[:, -1] == result_set.iloc[:, -2]).all() # all()只返回单个值
        # 更新数据
        if clusterChanged:
            # 重新计算质心
            cent_df = result_set.groupby(n+1).mean()
            centroids = cent_df.iloc[:, :n-1].values
            result_set.iloc[:, -1] = result_set.iloc[:, -2] # 更新质心索引
        # 画出每次迭代的图
        plt.scatter(result_set.iloc[:, 0], result_set.iloc[:, 1], c=result_set.iloc[:, -1])
        plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=80)
        plt.show()
    return centroids, result_set

"""
函数功能：聚类学习曲线(有助于选取k值)
参数说明：
    dataSet: 原始数据集
    cluster: kmeans聚类方法
    k: 簇的个数
返回：误差平方和SSE
"""
def kcLearningCurve(dataSet, cluster = kMeans, k=10):
    n = dataSet.shape[1]
    SSE = []
    for i in range(1, k):
        centroids, result_set = cluster(dataSet, i+1)
        SSE.append(result_set.iloc[:,n].sum())
    plt.plot(range(2, k+1), SSE, '--o')
    plt.show()
    return SSE

# -------------------------------------------
# 鸢尾花数据集测试

# 导入数据集
iris = pd.read_csv('iris.txt', header = None)
# print iris

# 生成随机质心
# iris_cent = randCent(iris, 3)
# print iris_cent

# iris_cent, iris_result = kMeans(iris, 3)
# print iris_result
# print iris_result.iloc[:,-1].value_counts()
# -------------------------------------------
# 测试数据集

testSet = pd.read_csv('testSet.txt', header=None, sep='\t')
# print testSet
# plt.scatter(testSet.iloc[:, 0], testSet.iloc[:, 1])
# plt.show()

ze = pd.DataFrame(np.zeros(testSet.shape[0]).reshape(-1, 1))
test_set = pd.concat([testSet, ze], axis=1, ignore_index = True)
# print test_set
test_cent, test_cluster = kMeans_l(test_set, 5)
print test_cluster
# # 画图
# plt.scatter(test_cluster.iloc[:, 0], test_cluster.iloc[:, 1], c=test_cluster.iloc[:, -1])
# plt.scatter(test_cent[:, 0], test_cent[:, 1], color='red', marker='x', s=80)
# plt.show()

# np.random.seed(123)
# for i in range(1, 5):
#     plt.subplot(2, 2, i)
#     test_cent, test_cluster = kMeans(test_set, 5)
#     plt.scatter(test_cluster.iloc[:, 0], test_cluster.iloc[:, 1], c=test_cluster.iloc[:, -1])
#     plt.plot(test_cent[:, 0], test_cent[:, 1], 'o', color='red')
#     print(test_cluster.iloc[:, 3].sum())
# plt.show()


# -------------------------------------------
# 计算误差平方和
# print test_cluster.iloc[:, 3].sum()
#
# kcLearningCurve(iris)
# kcLearningCurve(testSet)
# -------------------------------------------