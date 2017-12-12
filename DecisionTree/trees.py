#!/usr/bin/env python
# encoding: utf-8

"""
构建决策树
@author: caylaxu
@contact: xmy306538517@foxmail.com
@time: 17-12-12 下午4:35
"""
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {}
    # 1. 为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        # 2. 以2为底求对数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet:带划分的数据集
    :param axis:划分数据集的特征
    :param value:需要返回的特征的值
    :return:
    """
    # 1. 创建新的list对象
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 2. 抽取
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):

        # 1. 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0

        # 2. 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            # 3. 计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def createTree(dataSet, labels):
    """
    递归构建决策树
    :param dataSet:
    :param labels:
    :return:
    """
    classList = [example[-1] for example in dataSet]
    # 1. 类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 2. 遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}

    # 3. 得到列表包含的所有属性值
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    使用决策树的分类函数
    :param inputTree:
    :param featLabels:
    :param testVec:
    :return:
    """
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 1. 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    myDat, labels = createDataSet()
    # myDat[0][-1] = 'maybe'
    # print(myDat)
    # calcShannonEnt(myDat)

    # result1 = splitDataSet(myDat, 0, 1)
    # print result1
    # result2 = splitDataSet(myDat, 0, 0)
    # print result2

    # result = chooseBestFeatureToSplit(myDat)
    # print result

    myTree = createTree(myDat, labels)
    myDat, labels = createDataSet()# python引用问题 createTree时label已被修改
    result = classify(myTree, labels, [1, 1])
    print result
