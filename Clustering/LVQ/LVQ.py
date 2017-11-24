# -*- coding: utf-8 -*-
# @Time    : 2017/11/24 14:24
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : LVQ.py
# @Version : Python 3.6

'''
因为每次写算法都需要多写一步数据的预处理，浪费了许多时间，遂决定减少这部分的时间，然后以实现算法为主
'''

import numpy as np
import pandas as pd
import random

# LVQ 算法是假设样本是带有类别标记的（即Y）值，学习过程利用样本的这些监督信息来辅助聚类
class LVQ():

    # 因为LVQ需要初始化原始向量，k为簇数量, 最后返回原形向量， （这个算法有点类似于找质心的感觉）
    def StepbyStepLVQ(self,rowdata_x,rowdata_y,learning_rate,k):

        n_samples = rowdata_x.shape[0]
        n_features = rowdata_x.shape[1]

        # 初始化原形向量，这里做简单操作，假设希望找到 3个好瓜的簇，2个坏瓜的簇，总共五个簇
        good = np.where(rowdata_y == 1)[0]
        bad = np.where(rowdata_y == 0)[0]
        vectorlist = random.sample(list(good),3) + random.sample(list(bad),2)

        P_x = np.zeros((k,rowdata_x.shape[1]))
        P_y = np.zeros((k,rowdata_y.shape[1]))

        for i in range(len(vectorlist)):
            P_x[i] = rowdata_x[vectorlist[i]]
            P_y[i] = rowdata_y[vectorlist[i]]

        pp = 0
        # 迭代五十次
        while pp < 50:

            randomhang = random.randint(0,n_samples-1)
            random_X = rowdata_x[randomhang].reshape(1,n_features)

            minHang = self.__FindMinDist__(P_x,random_X)

            if rowdata_y[randomhang] == P_y[minHang]:
                P_x[minHang] = P_x[minHang] + learning_rate * (random_X - P_x[minHang])
            else:
                P_x[minHang] = P_x[minHang] - learning_rate * (random_X - P_x[minHang])

            pp = pp + 1

        print(P_x)
        return P_x


    # 计算方法，传入一个 P 原形 matrix ， 和一个随机的 X , 返回最小 P 向量的行向量数
    def __FindMinDist__(self,P,X):

        distM = np.sqrt(np.power((P - X), 2).sum(axis=1))
        return np.where(distM == np.min(distM))[0][0]


if __name__ == "__main__":

    # 数据为 周志华书中的 p202 表9.1 的西瓜数据
    mydata = pd.read_csv('./data/watermelon_data.txt', sep=',', encoding='utf-8', header=None)

    # 获取数据，前两列为输入样本数据，后面一列为标注数据
    rowdata_x = np.array(mydata[list(range(2))])
    rowdata_y = np.array(mydata[[2]])

    # 实例化类
    '''
        这里返回的是已经迭代了50后的原形变量，之后再将原始数据对他做最近距离比较，划分进相应的区域内，因为这部分比较简单这里就不进行
        代码赘述了。
    '''
    lvq = LVQ()
    lvq.StepbyStepLVQ(rowdata_x,rowdata_y,0.1,5)