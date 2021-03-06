#!/usr/bin/env python
# encoding: utf-8

"""

@author: caylaxu
@contact: xmy306538517@foxmail.com
@time: 17-12-12 下午8:26
"""
from numpy import *
import pprint

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    """
    词表到向量的转换函数
    :param dataSet:
    :return:
    """
    # 1 创建一个空集
    vocabSet = set([])
    for document in dataSet:
        # 2 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    词表到向量的转换函数
    :param vocabList:
    :param inputSet:
    :return:
    """
    # 3 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec


# def trainNB0(trainMatrix, trainCategory):
#     """
#     朴素贝叶斯分类器训练函数
#     :param trainMatrix: 文档矩阵
#     :param trainCategory:每篇文档类别所构成的向量
#     :return:
#     """
#     numTrainDocs = len(trainMatrix)
#     numWords = len(trainMatrix[0])
#     pAbusive = sum(trainCategory) / float(numTrainDocs)
#     # 1 初始化概率
#     p0Num = zeros(numWords)
#     p1Num = zeros(numWords)
#     p0Denom = 0.0
#     p1Denom = 0.0
#     for i in range(numTrainDocs):
#         if trainCategory[i] == 1:
#             # 2 向量相加
#             p1Num += trainMatrix[i]
#             p1Denom += sum(trainMatrix[i])
#         else:
#             p0Num += trainMatrix[i]
#             p0Denom += sum(trainMatrix[i])
#     # 对每个元素做除法
#     p1Vect = p1Num / p1Denom
#     p0Vect = p0Num / p0Denom
#     return p0Vect, p1Vect, pAbusive

def trainNB0(trainMatrix, trainCategory):
    """
    根据现实情况修改分类器后的朴素贝叶斯分类器训练函数
    :param trainMatrix: 文档矩阵
    :param trainCategory:每篇文档类别所构成的向量
    :return:
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 1 将所有词的出现数初始化为1,并将分母初始化为2
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 2 向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 通过求对数可以避免下溢出或者浮点数舍入导致的错误
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    :param vec2Classify: 要分类的向量
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    """
    # 1. 元素相乘
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postingDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print  testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


if __name__ == '__main__':
    # listOPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    #
    # trainMat = []
    # for postingDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(pAb)
    # print(p0V)
    # print(p1V)

    testingNB()
