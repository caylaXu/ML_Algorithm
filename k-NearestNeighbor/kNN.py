#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: caylaxu
@contact: caylaxu@fast.do
@software: PyCharm Community Edition
@file: kNN.py
@time: 17-12-11 下午8:11
"""
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    k-近邻算法
    :param inX:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    """
    dataSetSize = dataSet.shape[0]

    # 1. 距离计算(欧氏距离公式)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]

        # 2. 选择距离最小的k个点
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 3. 排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]


# ----------------------------------------------------------------------------------------
def file2matrix(filename):
    """
    将文本记录转换为NumPy的解析程序
    :param filename:
    :return:
    """
    fr = open(filename)
    arrayOLines = fr.readline()
    # 1. 得到文件的行数
    numberOfLines = len(arrayOLines)
    # 2. 创建返回的NumPy矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 3. 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    归一化特征值
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    #1. 特征值相除
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

if __name__ == '__main__':
    group, labels = createDataSet()
    result = classify0([0, 0], group, labels, 3)
    print(result)

