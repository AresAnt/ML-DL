# -*- coding: utf-8 -*-
# @Time    : 2017/11/23 15:46
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : Agnes.py
# @Version : Python 3.6

'''
因为每次写算法都需要多写一步数据的预处理，浪费了许多时间，遂决定减少这部分的时间，然后以实现算法为主
'''

import numpy as np
import pandas as pd

# AGNES 的算法是采用 自底向上的聚合策略的层次聚类算法。它首先将所有的数据当做单个簇然后找出，相邻的两个簇进行合并
class Agnes():

    # Ci 与 Cj ，分别为簇向量矩阵
    def __DistforClustering__(self,Ci,Cj,distFunction):

        ni_loop = Ci.shape[0]
        nj_loop = Cj.shape[0]

        # 全连锁
        if distFunction == 'dmax':

            min_dist = 0.0

            # 因为簇的数量不同所以可能会产生不同数据，以防万一做双重计算
            for j in range(nj_loop):
                distM = np.sqrt(np.power((Ci - Cj[j]), 2).sum(axis=1)).max()
                if min_dist < distM:
                    min_dist = distM

            for i in range(ni_loop):
                distM = np.sqrt(np.power((Cj - Ci[i]), 2).sum(axis=1)).max()
                if min_dist < distM:
                    min_dist = distM

        # 单连锁
        elif distFunction == 'dmin':

            min_dist = float("inf")

            # 因为簇的数量不同所以可能会产生不同数据，以防万一做双重计算
            for j in range(nj_loop):
                distM = np.sqrt(np.power((Ci - Cj[j]), 2).sum(axis=1)).min()
                if distM < min_dist:
                    min_dist = distM

            for i in range(ni_loop):
                distM = np.sqrt(np.power((Cj - Ci[i]), 2).sum(axis=1)).min()
                if distM < min_dist:
                    min_dist = distM

        # 平均连锁
        elif distFunction == 'davg':

            min_dist = 0.0

            # 因为簇的数量不同所以可能会产生不同数据，以防万一做双重计算
            for j in range(nj_loop):
                min_dist = min_dist + (np.sqrt(np.power((Ci - Cj[j]), 2).sum(axis=1)).sum()) / (ni_loop + nj_loop)

        return min_dist



    # 开始按照最简单的方式一步一步来进行操作
    def StepbyStepAGNES(self,rowdata,k):

        n_samples = rowdata.shape[0]
        Clustering = {}

        # 初始化簇类，将每个样本都标记为一个簇
        for i in range(n_samples):
            Clustering["cl"+str(i)] = rowdata[i].reshape(1,rowdata.shape[1])

        M = np.zeros((n_samples,n_samples),dtype=float)

        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    M[i][j] = float("inf")
                    continue
                M[i][j] = self.__DistforClustering__(Clustering["cl"+str(i)],Clustering["cl"+str(j)],"dmax")

        q = n_samples
        while q > k:

            FindNearClustering = np.where(M == np.amin(M))
            tempClustering = {}
            j = 0
            for i in range(q-1):
                if i == FindNearClustering[0][0]:
                    tempClustering["cl"+str(i)] = np.row_stack((Clustering["cl"+str(j)],Clustering["cl"+str(FindNearClustering[0][1])]))
                    j = j+1
                    continue
                elif i == FindNearClustering[0][1]:
                    j = j + 1

                tempClustering["cl"+str(i)] = Clustering["cl"+str(j)]
                j = j + 1
            Clustering = tempClustering

            M = np.zeros((q-1,q-1),dtype=float)

            for i in range(q-1):
                for j in range(q-1):
                    if i == j:
                        M[i][j] = float("inf")
                        continue
                    M[i][j] = self.__DistforClustering__(Clustering["cl" + str(i)], Clustering["cl" + str(j)], "dmax")
            q = q - 1


        cls = 1
        for k,v in Clustering.items():
            print("Clustering"+str(cls)+":")
            print(set([x+1 for x in np.where(list(x.all() for x in np.isin(rowdata,v)))[0]]))
            cls = cls+1

if __name__ == "__main__":

    # 数据为 周志华书中的 p202 表9.1 的西瓜数据
    mydata = pd.read_csv('./data/watermelon_data.txt', sep=',', encoding='utf-8', header=None)

    # 获取前两列数据，最后一列为分类标注的数据，可以在聚类中暂时忽略
    rowdata = np.array(mydata[list(range(2))])

    # 类实例化
    Ag = Agnes()
    Ag.StepbyStepAGNES(rowdata,7)
