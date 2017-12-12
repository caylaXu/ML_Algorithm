#!/usr/bin/env python
# encoding: utf-8

"""
使用kNN改进约会网站的配对效果
@author: caylaxu
@contact: caylaxu@fast.do
@time: 17-12-12 上午8:11
"""
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


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


# -----------------------------------------------------------------------------------------------------------------------

def file2matrix(filename):
    """
    将文本记录转换为NumPy的解析程序
    :param filename:
    :return:
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
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
    归一化特征值（处理成0到1或者-1到1之间）公式newValue = (olsValue - min)/(max -min)
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 1. 特征值相除
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    分类器针对约会网站的测试代码
    :return:
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with : %d, the real answer is :%d " % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is : %f" % (errorCount / float(numTestVecs))


def classifyPerson():
    """
    约会网站预测函数
    :return:
    """
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", resultList[classifierResult - 1]


if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print(datingDataMat)
    print(datingLabels[0:20])

    # 输出散点图，直观查看
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.xlabel('percentage of time spent playing video games')
    # plt.ylabel('liters of ice cream consumed per year')
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # plt.show()

    normMat, ranges, minVals = autoNorm(datingDataMat)
    print(normMat)
    print(ranges)
    print(minVals)
    datingClassTest()
    classifyPerson()


