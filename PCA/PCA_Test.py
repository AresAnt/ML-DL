# -*- coding: utf-8 -*-
# @Time    : 2017/11/7 14:39
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : PCA_Test.py
# @Software: PyCharm

import sys
from PCA import Simple_PCA

if __name__ == "__main__":
    # 类实例化
    PcaClass = Simple_PCA.Simple_PCA()

    # 测试可以调用 PCA 类中内容
    PcaClass.call_PCA()

    # 路径
    dataFile = './iris.data.txt'

    # 查看原始数据 第一维与第二维的数据图像, loadData 函数有两个参数，第二个参数为分隔符
    rawData = PcaClass.load_data(dataFile,',')
    PcaClass.DrawPic(rawData,2,3)

    # 查看PCA后（正则化以及扩展到原始维度进行样本比较,即观察降维后的模型）
    PCA_Data,PCA_Pic_Data = PcaClass.StepByStep_PCA(rawData,3)
    PcaClass.DrawPic(PCA_Pic_Data,2,3)