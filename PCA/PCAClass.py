# -*- coding: utf-8 -*-
# @Time    : 2017/11/7 14:38
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : PCAClass.py
# @Software: PyCharm


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

class PCA():

    # 初始化
    def __init__(self):
        pass

    # 测试是否被Test所调用
    def call_PCA(self):
        print("Hello World!")

    # 通过传入数据路径来读取数据类型
    def load_data(self,filepath,seprate='\t'):
        mydata = pd.read_csv(filepath,sep=seprate,encoding='utf-8',header=None)
        #mydata.columns = [str(k+1) for k in range(mydata.shape[1])]

        return np.mat(mydata)

    # 通过PCA规则，从而一步一步的实现
    def StepByStep_PCA(self,data,featureVector=9999999):
        nor_data = self.__normalization__(data)

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

    # 中心化 = 归零化 + 平常化  公式 x' = ( x - μ ) / σ   μ 表示均值， σ 表示标准差
    def __normalization__(self,datas):
        # 样本数据每一列为一个样本 将样本数据的均值求出来后进行归零化
        average_Xm = np.mean(datas,axis=0)

        # 归零化
        average_Data = datas - average_Xm
        # print(average_Data)

        # 平常化，即 average_data / 标准差
        Normal_Xm = np.dot(average_Data.T,average_Data)/ (average_Data.shape[0] - 1)

        # 这边shape[1] 指的商榷，如果数据格式是二维格式的话是可以的获取用多少列 及 多少样本量
        num_Col = Normal_Xm.shape[1]
        for i in range(num_Col):
            sum_squares = np.sqrt(Normal_Xm[i,i])
            # print("sum:",sum_squares)
            average_Data[:,i] = average_Data[:,i] / sum_squares

        return average_Data

    # 图像显示来，因为是二维图像，用来看某两个特征的二维关系图
    def DrawPic(self,data,feature_1,feature_2):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(data[:,feature_1].flatten().A[0],data[:,feature_2].flatten().A[0],marker='o',c='y',s=50)
        plt.show()

