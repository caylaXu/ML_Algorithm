#!/usr/bin/env python
# encoding: utf-8

"""
使用皮素贝叶斯分类器从个人广告中获取区域倾向
@author: caylaxu
@contact: xmy306538517@foxmail.com
@time: 17/12/14 08:23
"""

import feedparser
import operator


def calcMostFreq(vocabList, fullText):
    """
    计算出现频率
    :param vocabList:
    :param fullText:
    :return:
    """
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    # 每次访问一条RSS源
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)

    @todo没写完呢


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
