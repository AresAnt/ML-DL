# -*- coding: utf-8 -*-
# @Time    : 2017/11/13 15:58
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : KNN_Test.py
# @Version : Python 3.6

from KNN import Data_Processing
from KNN import KNN_Class
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # 类实例化
    DP = Data_Processing.Data_Process()
    KNN = KNN_Class.KNN_Class()

    # 数据路径
    filepath = './data/datingTestSet.txt'
    xdata,ydata = DP.load_data(filepath)

    # 划分训练集与测试集
    x_train,x_test,y_train,y_test = train_test_split(xdata,ydata,test_size=0.1,random_state=7)

    # 进行测试集测试
    rate = 0.0
    for i in range(x_test.shape[0]):
        predict_y = KNN.StepbyStepKNN(x_train,y_train,x_test[i],5)
        real_y = int(y_test[i][0])
        if predict_y == real_y:
            rate = rate + 1
        print("The classifier is: ",str(predict_y),", the real answer is: ",str(real_y))

    print("accuracy is: ",rate/y_test.shape[0])