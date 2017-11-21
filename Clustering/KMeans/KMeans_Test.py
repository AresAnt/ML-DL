# -*- coding: utf-8 -*-
# @Time    : 2017/11/17 21:41
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : KMeans_Test.py
# @Version : Python 3.6

from Clustering.KMeans import Data_Processing
from Clustering.KMeans import KMeans_Class
import numpy as np

if __name__ == '__main__':

    # 类实例化
    DP = Data_Processing.Data_Process()
    KM_Simple = KMeans_Class.KMeans()

    # 读取数据
    filepath = './data/testSet.txt'
    rowData = DP.load_data(filepath)

    # 进行操作
    KM_Simple.StepByStepKMeans(rowData,4)
    DP.DrawPic(np.mat(rowData),0,1)
