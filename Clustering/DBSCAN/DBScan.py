# -*- coding: utf-8 -*-
# @Time    : 2017/11/26 13:38
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : DBScan.py
# @Version : Python 3.6

import numpy as np
import pandas as pd
import random
import queue

class DBScan():

    def StepbyStepDBSCAN(self,X_Array,threshold,MinPts):

        # 初始化核心对象集
        n_samples = X_Array.shape[0]
        n_features = X_Array.shape[1]

        CoreList = []
        for i in range(n_samples):
            if self.__FindCodeSeed__(X_Array,X_Array[i],threshold,MinPts):
                CoreList.append(i)

        print(CoreList)

        k = 0
        T = rowdata.copy()
        Told = T.copy()

        # 用来存放下面循环时候每次从总簇中去除掉的簇
        tempSet = set()

        while CoreList:
            item = random.sample(CoreList,1)[0]

            # 建立FIFO的队列
            QList = queue.Queue()
            QList.put(item)

            # 把加入队列的取出，然后再矩阵中，将其无穷大，这样子就可以在计算时候被排除
            T[item] = np.full((1,n_features),np.inf)

            while not QList.empty():
                tempitem = QList.get()

                if self.__FindCodeSeed__(Told,Told[tempitem],threshold,MinPts):
                    # 找到当前节点下的密度直达
                    ddrlist = self.__FindCodeList__(T,Told[tempitem],threshold)
                    for pp in ddrlist:
                        QList.put(pp)
                        T[pp] = np.full((1,n_features),np.inf)

            # dc 变量表示 每次与原始数据有何变化，即每次多余出来的变化为更新出来的新簇
            dc = np.where([x.all() for x in (Told != T)])[0]

            # sc 变量表示 每次变化后还剩余了多少元素
            sc = np.where([x.all() for x in (Told == T)])[0]

            # 输出每次聚成的簇
            k = k + 1
            print("C"+str(k)+":",[ cl for cl in (set(dc) - set(tempSet))])

            tempSet = set(dc)

            CoreList = list(set(sc) & set(CoreList))




    # 判断这个种子是否符合期望，能够成为期望种子
    def __FindCodeSeed__(self,RowArray,core,threshold,MinPts):
        distM = np.sqrt(np.power((RowArray - core), 2).sum(axis=1))
        return (MinPts<= len(np.where(distM<=threshold)[0]))

    # 找到这个期望种子的领域集合
    def __FindCodeList__(self,T,core,threshold):
        distM = np.sqrt(np.power((T - core), 2).sum(axis=1))
        return(np.where(distM<=threshold)[0])

if __name__ == "__main__":

    # 数据为 周志华书中的 p202 表9.1 的西瓜数据
    mydata = pd.read_csv('./data/watermelon_data.txt', sep=',', encoding='utf-8', header=None)

    # 获取前两列数据，最后一列为分类标注的数据，可以在聚类中暂时忽略
    rowdata = np.array(mydata[list(range(2))])

    # 类实例化
    dbs = DBScan()
    '''
        出来的结果可能与周志华书上有时候会有一些不一样，这是因为每次初始的随机核心对象不同。
        比如 4，25 的数据是符合规定的，但是它们在书上却分在了不同的区域。
        11 与 15号数据，不符合分类将不会被划分进簇组内
    '''
    dbs.StepbyStepDBSCAN(rowdata,0.11,5)