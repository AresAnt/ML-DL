# -*- coding: utf-8 -*-
# @Time    : 2017/11/13 14:50
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : Data_Processing.py
# @Version : Python 3.6

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class Data_Process():

    # 通过传入数据路径来读取数据类型, 有监督数据
    def load_data(self, filepath, seprate='\t'):
        mydata = pd.read_csv(filepath, sep=seprate, encoding='utf-8', header=None)
        return np.array(mydata)

    # 图像显示来，因为是二维图像，用来看某两个特征的二维关系图
    def DrawPic(self,data,feature_1,feature_2):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(data[:,feature_1].flatten().A[0],data[:,feature_2].flatten().A[0],marker='o',c='y',s=50)
        plt.show()
