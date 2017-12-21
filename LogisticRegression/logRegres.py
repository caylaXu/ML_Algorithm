#!/usr/bin/env python
# encoding: utf-8

"""
Logistic 回归梯度上升优化算法、Logistic 回归随机梯度上升优化算法
@author: caylaxu
@contact: xmy306538517@foxmail.com
@time: 17-12-21 下午8:05
"""
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """
    Logistic 回归梯度上升优化算法
    :param dataMatIn:
    :param classLabels:
    :return:
    """
    # 转换为NumPy矩阵数据类型
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001  # alpha是向目标移动的步长
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        # 矩阵相乘
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):
    """
    画出数据集和Logistic回归最佳拟合直线的函数
    :param wei:
    :return:
    """
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 最佳拟合直线 设定0 = w 0 x 0 + w 1 x 1 + w 2 x 2 ,然后解出X2和X1的关系式(即分隔线的方程,注意X0=1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.xlabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度下降
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    alpha = 0.1
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # alpha每次迭代时需要调整，岁迭代次数减小，但不会小到0
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机选取更新
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights

if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # weights = gradAscent(dataArr, labelMat)
    # print(weights)
    # plotBestFit(weights.getA())

    # weights = stocGradAscent0(dataArr, labelMat)
    # print(weights)
    # plotBestFit(weights)

    weights = stocGradAscent1(array(dataArr), labelMat)
    print(weights)
    plotBestFit(weights)
