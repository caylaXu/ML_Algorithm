#!/usr/bin/env python
# encoding: utf-8

"""
使用决策树预测隐形眼镜类型
@author: caylaxu
@contact: xmy306538517@foxmail.com
@time: 17-12-12 下午7:54
"""
import trees

if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = trees.createTree(lenses, lensesLabels)
    print lensesTree