# -*- coding: utf-8 -*-
# @Time    : 2017/11/7 14:38
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : Simple_PCA.py
# @Version : Python 3.6


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class Simple_PCA():

    # 初始化
    def __init__(self):
        pass

    # 测试是否被Test所调用
    def call_PCA(self):
        print("Call Simple PCA!")

    # 通过PCA规则，从而一步一步的实现
    def StepByStep_PCA(self,data,featureVector=9999999):
        nor_data = data

        # 计算协方差 rowvar 来标明样本m是以行计算，还是列计算，默认（True）是行计算
        Covariation = np.cov(nor_data,rowvar=False)

        # 计算出特征值与特征向量
        eigVals,eigVects = np.linalg.eig(np.mat(Covariation))

        # 将特征值的大小，从小到大进行排序
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[: -(featureVector+1):-1]
        redEigVects = eigVects[:,eigValInd]

        # 降维后的数据
        lowDataMat = nor_data * redEigVects

        lowData2Dimension = lowDataMat * redEigVects.T

        return lowDataMat,lowData2Dimension

