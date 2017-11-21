# -*- coding: utf-8 -*-
# @Time    : 2017/11/17 21:32
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : KMeans_Class.py
# @Version : Python 3.6

import numpy as np
import random

class KMeans():

    # 传入想要分割的 K类 以及数据源
    def StepByStepKMeans(self,rowdata,k):

        # 用一个字典存放聚类后不同的簇标记
        Clustering = {}

        # 随机选择样本中的 4 个样本作为初始均值向量
        num_samples = rowdata.shape[0]
        centroid = np.zeros((k,rowdata.shape[1]))
        for i in range(k):
            centroid[i,:] = rowdata[random.randint(0,num_samples-1)]
            Clustering["cl"+str(i+1)] = []

        # 默认循环十次来逼近距离
        n = 0
        while n < 10:

            for i in range(num_samples):
                close_index = self.__DistEClud__(rowdata[i],centroid)
                Clustering["cl"+str(close_index + 1)].append(i)

            tempcentroid = np.copy(centroid)

            # 对质心进行更新，筛选出来的类进行
            for j in range(k):
                clussterAssment = Clustering["cl"+str(j+1)]
                Clustering["cl" + str(j + 1)] = []
                if len(clussterAssment) == 0:
                    centroid[j,:] = centroid[j,:]
                else:
                    tempdata = rowdata[clussterAssment[:]].mean(axis=0).reshape(1,centroid.shape[1])
                    centroid[j,:] = tempdata

            if (centroid - tempcentroid).sum() == 0:
                break

            print(centroid - tempcentroid)
            n = n+1

    def __DistEClud__(self,vecA,vecB):
        ABDist = np.sqrt((np.power(vecA-vecB,2)).sum(axis=1))
        maxIndex = np.where(ABDist == ABDist.min())[0][0]
        return maxIndex
