#!/usr/bin/env python
# encoding: utf-8

"""
马疝病数据集上应用AdaBoost
@author: caylaxu
@Time: 18-2-6 下午9:27
"""
import adaboost
from numpy import*


def loadDataSet(fileName):
    """
    自适应数据加载函数
    :param fileName:
    :return:
    """
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifilerArray = adaboost.adaBoostTrainDS(dataArr, labelArr, 10)

    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaboost.adaClassify(testArr,classifilerArray)
    errArr = mat(ones((67, 1)))
    print(errArr[prediction10 != mat(testLabelArr).T].sum())