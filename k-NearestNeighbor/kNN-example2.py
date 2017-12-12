#!/usr/bin/env python
# encoding: utf-8

"""
使用k-近邻算法的手写识别系统
@author: caylaxu
@contact: xmy306538517@foxmail.com
@time: 17-12-12 下午1:59
"""

from numpy import *
import operator
from os import listdir


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

def img2Vector(filename):
    """
    准备数据：将图像转换为测试向量
    :param filename:
    :return:
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    """
    手写数字识别系统的测试代码
    :return:
    """
    hwLabels = []
    # 获取目录内容
    trainningFileList = listdir('trainingDigits')
    m = len(trainningFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 从文件名解析分类数字
        fileNameStr = trainningFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = float(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2Vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with : %d, the real answer is: %d" % (classifierResult, classNumStr)

        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthr total error rate is: %f" % (errorCount/float(mTest))

if __name__ == '__main__':
    handwritingClassTest()

