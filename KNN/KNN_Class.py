# -*- coding: utf-8 -*-
# @Time    : 2017/11/13 14:51
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : KNN_Class.py
# @Version : Python 3.6

import numpy as np

class KNN_Class():

    # 监督学习，先输入已经标称好的数据，然后通过输入预测点，做全部的欧氏距离计算，然后排序后按一定范围进行筛选按比例来选择分类结果
    def StepbyStepKNN(self,x_train,y_train,x_test,K):
        dataSize = x_train.shape[0]

        # 矩阵相减
        diffMat = x_train - np.tile(x_test,(dataSize,1))

        # 因为之前Process整理出来的是 matrix , matrix 与 array 有计算上的不同， 比如 **2 matrix是内积
        diffMat = np.array(diffMat)

        # 算出欧式距离，然后排序进行寻找最短的 K 个值
        sqDistance = np.sqrt((diffMat**2).sum(axis=1))
        sqDistance = sqDistance.reshape(sqDistance.shape[0],1)

        KInd = np.argsort(sqDistance,axis=0)
        classCount = {}
        for i in range(K):
            co_Y = y_train[KInd[i]]
            co_Y = co_Y[0][0]
            classCount[co_Y] = classCount.get(co_Y,0) + 1
        sortedClass = sorted(classCount.items(),key=lambda item:item[1],reverse=True)
        return int(sortedClass[0][0])
