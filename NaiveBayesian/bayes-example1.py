#!/usr/bin/env python
# encoding: utf-8

"""
使用朴素贝叶斯过滤垃圾邮件
@author: caylaxu
@contact: xmy306538517@foxmail.com
@time: 17/12/13 08:13
"""

import re
import bayes
import random
from numpy import *
import pprint


def bagOfWords2VecMN(vocabList, inputSet):
    """
    朴素贝叶斯词袋模型
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):  # input is big string, #output is word list
    """
    接收一个大字符串并将其解析成为字符串列表
    :param bigString:
    :return:
    """
    # 去除标点符号
    listOfTokens = re.split(r'\W*', bigString)
    # 统一大小写 去空字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    """
    对贝叶斯垃圾邮件分类器进行自动化处理
    :return:
    """
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 1. 导入并解析文本文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        # 2. 随机构建训练集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    # 3. 对测试集分类
    for docIndex in testSet:
        wordVector = bayes.setOfWords2Vec(vocabList, docList[docIndex])
        if bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)


if __name__ == '__main__':
    spamTest()
    pass
