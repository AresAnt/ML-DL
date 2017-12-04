# -*- coding: utf-8 -*-
# @Time    : 2017/12/4 15:38
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : Mixture_Gaussian.py
# @Version : Python 3.6

import numpy as np
import pandas as pd
import math
import random

class MG():

    def StepbyStepMG(self,rowdata,k):

        n_samples = rowdata.shape[0]
        n_feature = rowdata.shape[1]

        repeate = 0

        # 用来记录簇标记更改的矩阵向量
        xj = np.zeros((n_samples,k))

        t_list = random.sample(range(n_samples), 3)

        # 初始化数据记录
        K_dict = {}

        # 用来记录分类
        Clustering = {}

        for i in range(k):
            temp_dict = {}
            temp_dict["a"] = 1 / k
            temp_dict["u"] = rowdata[t_list[i]].reshape(1,n_feature)
            temp_dict["Z"] = np.array([[0.1,0],[0,0.1]])
            K_dict["cl"+str(i)] = temp_dict
            Clustering["cl"+str(i+1)] = []

        # 进行迭代
        while repeate < 50:

            # E 步计算
            for i in range(n_samples):
                for j in range(k):
                    x_row = rowdata[i].reshape(1,n_feature)
                    xj[i][j] = self.__P_Gaussian__(x_row,K_dict["cl"+str(j)])

                total  = xj[i].sum()
                for j in range(k):
                    xj[i][j] = xj[i][j] / total

            # M 步计算
            for i in range(k):
                temp_dict = {}
                temp_dict["a"] = xj[:,i].mean()
                temp_dict["u"] = (np.dot(xj[:,i].T,rowdata).reshape(1,n_feature)) / (xj[:,i].sum())
                temp_dict["Z"] = self.__C_xiefangcha__(xj[:,i],rowdata,temp_dict["u"])
                K_dict["cl"+str(i)] = temp_dict

            repeate = repeate + 1

        for i in range(n_samples):
            cl = np.where(xj[i] == xj[i].max())[0][0]
            Clustering["cl"+str(cl+1)].append(i+1)

        print(Clustering)

    # 计算更新协方差矩阵
    def __C_xiefangcha__(self,rij,rowdata,u):
        n_samples = rij.shape[0]
        pp = np.zeros((u.shape[1],u.shape[1]))

        for i in range(n_samples):
            pp = pp + rij[i] * np.dot((rowdata[i] - u).T,(rowdata[i] - u))

        return pp / rij.sum()

    # 用高斯概率密度函数来计算 后验概率
    def __P_Gaussian__(self,x,K_dict):
        a = K_dict["a"]
        u = K_dict["u"]
        Z = K_dict["Z"]
        n = Z.shape[0]

        qian = 1 / (np.sqrt((2*math.pi) ** n) * np.linalg.det(Z))
        one = np.dot((x - u),np.linalg.inv(Z))
        hou = np.exp(-0.5 * np.dot(one,(x - u).T))
        return a * qian * hou

if __name__ == "__main__":

    # 数据为 周志华书中的 p202 表9.1 的西瓜数据
    mydata = pd.read_csv('./data/watermelon_data.txt', sep=',', encoding='utf-8', header=None)

    # 获取前两列数据，最后一列为分类标注的数据，可以在聚类中暂时忽略
    rowdata = np.array(mydata[list(range(2))])

    # 实例化
    mg = MG()
    mg.StepbyStepMG(rowdata,3)