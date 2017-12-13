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


def bagOfWords2VecMN(vocabList, inputSet):
    """
    朴素贝叶斯词袋模型
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):  # input is big string, #output is word list
    # 去除标点符号
    listOfTokens = re.split(r'\W*', bigString)
    # 统一大小写 去空字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 导入并解析文本文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    trainSet = range(50);
    testSet = []

    @todo没写完呢


if __name__ == '__main__':
    pass
