# -*- coding: utf-8 -*-
# @Time    : 2017/11/12 16:04
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : Kernel_PCA.py
# @Version : Python 3.6

import numpy as np
import math
from functools import reduce

# 高斯核函数的PCA非线性降维
class Kernel_PCA():

    # 初始化,并定义用何种核函数
    def __init__(self,Kernel_Function):
        self.Kernel = Kernel_Function

    # 测试是否被Test所调用
    def call_PCA(self):
        print("Call "+ self.Kernel + " Kernel PCA!")

    # 自行设计每步的核主成分分析
    def StepbyStep_KPCA(self,data):
        return self.__Kernel_Function__(data)

    # 选择核函数进行操作，这步可以个人补充
    def __Kernel_Function__(self,data):
        M = data.shape[0]
        K = np.zeros((M, M), dtype=float)

        for i in range(M):
            for j in range(M):
                if self.Kernel == 'Gaussian':
                    K[i,j] = self.__Gaussian_Kernel__(data[i,:],data[j,:],1.0)

        # 核矩阵的修正
        ones = np.ones((M, M), dtype=float) / M
        K = K - np.dot(ones,K) - np.dot(K,ones) + np.dot(np.dot(ones, K),ones)

        # 计算出特征值与特征向量
        eigVals, eigVects = np.linalg.eig(np.mat(K))

        # 将特征值的大小，从小到大进行排序，即可求出对应的 α 向量
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[: :-1]
        redEigVects = eigVects[:, eigValInd]

        # 施密特正交化对 α 操作
        for i in range(len(eigValInd)):
            redEigVects[:,i] = redEigVects[:,i] / math.sqrt(abs(eigVals[eigValInd[i]] + 1.e-18))

        # 计算累计贡献率,进行筛选出前面吉祥
        Percent = 0.0
        temp = 0.0
        Total = reduce(lambda x,y:x+y, eigVals)
        print(eigVals)
        for i in range(len(eigValInd)):
            temp = temp + eigVals[eigValInd[i]]
            Percent = temp / Total
            print("Percent:",Percent)
            if Percent > 0.9:
                break

        redEigVects = redEigVects[:,0:i+1]

        # 返回投影后的坐标
        return K*redEigVects

    # 高斯核函数 k(x,x') = exp(-||x - x'||² / C)
    def __Gaussian_Kernel__(self,x,y,C):
        return math.exp(-(np.linalg.norm((x - y), ord=2) ** 2) / C)


