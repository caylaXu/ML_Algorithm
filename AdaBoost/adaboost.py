#!/usr/bin/env python
# encoding: utf-8

"""
基于单层决策树构建弱分类器
@author: caylaxu
@contact: xmy306538517@foxmail.com
@time: 18-1-5 下午9:36
"""
from numpy import *


def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


###############基于单层决策树构建弱分类器################
"""
将最小错误率 minError 设为+∞
对数据集中的每一个特征(第一层循环):
    对每个步长(第二层循环):
        对每个不等号(第三层循环):
            建立一棵单层决策树并利用加权数据集对它进行测试
            如果错误率低于 minError ,则将当前单层决策树设为最佳单层决策树
返回最佳单层决策树
"""


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # 1. 计算加权错误率
                weightedError = D.T * errArr
                print "split: dim %d, thresh %.2f, thresh ineqal:\
                        %s, the weighted error is %.3f" % \
                      (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


##########完整 AdaBoost 算法的实现##########
"""
对每次迭代:
    利用 buildStump() 函数找到最佳的单层决策树
    将最佳单层决策树加入到单层决策树数组
    计算alpha
    计算新的权重向量 D
    更新累计类别估计值
    如果错误率等于0.0,则退出循环
"""


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D:", D.T
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst:", classEst.T
        # 1. 为下一次迭代计算D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # 2. 错误率累计
        aggClassEst += alpha * classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate, "\n"
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(datToClass, classifierArr):
    """
    AdaBoost分类函数
    :param datToClass:
    :param classifierArr:
    :return:
    """
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)


if __name__ == '__main__':
    datMat, classLabels = loadSimpData()
    # test1 基于单层决策树的弱分类器
    # D = mat(ones((5, 1)) / 5)
    # buildStump(datMat, classLabels, D)

    # test2 基于单层决策树的AdaBoost分类器
    classifierArr = adaBoostTrainDS(datMat, classLabels, 30)
    adaClassify([0, 0], classifierArr)
