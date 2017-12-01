# -*- coding: utf-8 -*-
# @Time    : 2017/11/30 20:37
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : NavieBayes.py
# @Version : Python 3.6

import numpy as np
import pandas as pd
import math

class NavieBayes():

    # 假装开始传入数据开始训练
    def TrainningNavieBayes(self,rowdata_x,rowdata_y):
        rowdata_x = np.c_[rowdata_x,rowdata_y]

        self.x = rowdata_x
        self.y = {}

        '''
            查找出有多少种分类，并对其进行编号，进行先验概率的录入
            fla_y 表示分类的矩阵 list 化
            N_Y 表示 rowdata_Y 的总数量
            Y_N 表示有多少种 Y 的类型
        '''
        fla_y = list(rowdata_y.flatten())
        N_Y = len(fla_y)
        Y_N = len(set(fla_y))
        for c in set(fla_y):
            self.y["cl"+str(c)] = [fla_y.count(c),self.__Laplacian__(fla_y.count(c),N_Y,Y_N)]

    # 极大似然估计，这里使用了正太分布的操作，后续可以再改扩充
    def __MLE__(self,feature,cl,x_sample):
        Y_N = len(np.where(self.x[:,8] == cl)[0])
        cl_array = self.x[np.where(self.x[:,8] == cl)[0]]

        # 计算特征 i 的均值与方差
        u = cl_array[:,feature].mean()
        xita = np.sqrt(np.dot((cl_array[:,feature] - u),(cl_array[:,feature] - u).T) / Y_N)

        # print("Cl:",cl,"feature:",feature,"x_sample:",x_sample,"u:",u,"xita:",xita)
        # print((1 / (np.sqrt(2 * math.pi) * xita)) * np.exp(-((x_sample - u)**2) / (2 * (xita**2))))
        return (1 / (np.sqrt(2 * math.pi) * xita)) * np.exp(-((x_sample - u)**2) / (2 * (xita**2)))

    # 拉普拉斯平滑
    def __Laplacian__(self,dc,d,n):

        return float((dc+1)/(d+n))

    '''
        测试数据,传进来的test_x为一维数据
        这里采用了“懒惰学习”的方式，通过传进数据后再开始对数据进行计算。
        一般情况下为了预测速度比较快，可以再 Trainning 阶段设计一个矩阵表， 之后传进来的数据，只要查看矩阵表获取相对应的数据即可
    '''
    def TestingNavieBayes(self,test_x):

        test_y = {}

        # 因为有部分数据是要做极大似然估计，所以这里暂时区分开来
        n_features = len(test_x)

        for k,v in self.y.items():

            cl = float(k.replace("cl",''))
            Num_feature_cl = len(set(self.x[:, 8].flatten()))
            p_feature_cl = self.__Laplacian__(Num_feature_cl,v[1],Num_feature_cl)
            p_log = np.log(p_feature_cl)
            for i in range(n_features):
                feature = test_x[i]

                # 以后可以重写，判断数据为离散型或者连续型，进行区别对待
                if i == 6 or i == 7:
                    p_feature_cl = self.__MLE__(i,cl,feature)
                    p_log = p_log + np.log(p_feature_cl)
                else:
                    Num_feature_cl = len(set(self.x[:,i].flatten()))
                    # print(Num_feature_cl)
                    p_feature_cl = self.__Laplacian__(Num_feature_cl,v[0],Num_feature_cl)
                    p_log = p_log + np.log(p_feature_cl)
                    # print(p_log)

            test_y[k] = p_log

        sort_Y = sorted(test_y.items(),key=lambda v:v[1],reverse=True)

        return sort_Y[0][0]


if __name__ == "__main__":

    '''
    数据为 周志华书中的 p84 表4.3 的西瓜数据:

        青绿 = 1 ， 乌黑 = 2 ， 浅白 = 3
        蜷缩 = 1 ， 稍蜷 = 2 ， 硬挺 = 3
        浊响 = 1 ， 沉闷 = 2 ， 清脆 = 3
        清晰 = 1 ， 稍糊 = 2 ， 模糊 = 3
        凹陷 = 1 ， 稍凹 = 2 ， 平坦 = 3
        硬滑 = 1 ， 软粘 = 2
        好瓜 = 1 ， 坏瓜 = 0

    事先说明一点，这里并没有做智能化，因为还在学习，所以对于非离散型数据，密度和含糖率会单独用极大似然法来进行估计。
    '''
    mydata = pd.read_csv('./data/watermelon.txt', sep=',', encoding='utf-8', header=None)

    # 获取前两列数据，最后一列为分类标注的数据，可以在聚类中暂时忽略
    rowdata_x = np.array(mydata[list(range(8))])
    rowdata_y = np.array(mydata[[8]])

    # 实例化
    nb = NavieBayes()
    nb.TrainningNavieBayes(rowdata_x,rowdata_y)

    clustering = { "cl0":"坏瓜","cl1":"好瓜"}
    # cl = nb.TestingNavieBayes(rowdata_x[0])
    cl = nb.TestingNavieBayes(np.array([1,2,3,3,1,2,0.888,0.362]))
    print(clustering[cl])