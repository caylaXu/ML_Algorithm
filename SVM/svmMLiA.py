#!/usr/bin/env python
# encoding: utf-8

"""
SMO 高效优化算法
@author: caylaxu
@contact: xmy306538517@foxmail.com
@time: 17-12-22 上午11:29
"""
from numpy import *


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """

    :param i:alpha的下标
    :param m:alpha的数目
    :return:
    """
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            # 1. 如果alpha可以更改进入优化过程
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 2. 随机选择第二个alpha
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 3. 保证alpha在0到C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print "L==H"
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print "eta>=0"
                    continue
                alphas[i] -= labelMat[j] * [Ei - Ej] / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abd(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"
                    continue
                # 4. 对i进行修改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b=b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged +=1
                print "iter: %d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter +=1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return b,alphas


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    print(labelArr)
