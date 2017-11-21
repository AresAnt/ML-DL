# -*- coding: utf-8 -*-
# @Time    : 2017/11/15 17:13
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : Birch_Class.py
# @Version : Python 3.6

import numpy as np

# 建立CF节点，一个书的支点可以有 β 个 CF节点
class CFNode():

    # 获取初始参数，然后初始化节点
    def __init__(self,XArray):
        n_sample,n_features = XArray.shape
        self.N = 1
        self.LS = XArray
        self.ES = np.dot(XArray,XArray.T).sum()
        # print(self.N,self.LS,self.ES)

        # belong 说明当前这个 CFnode 属于何个 TreeNode, CFnode 的child 都指向一个 Treenode
        self.belong = None
        self.child = None

# 建立树节点
class TreeNode():

    # 获取初始参数，然后初始根节点
    def __init__(self,branch_balance,leaf_balance,threshold,is_leaf,n_features):
        self.branch_balance = branch_balance
        self.leaf_balance = leaf_balance
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.n_features = n_features

        # 初始化节点内容
        self.N = 0
        self.LS = np.zeros((1,n_features))
        self.ES = 0.0

        # treenodes 来表示每一个node节点里面有几个 CFnode, parent 都指向一个 CFnode
        self.treenodes = []
        self.parent = None

    # 往树节点中插入CF特征点
    def TreeNode_insert_CFNode(self,CFNode):
        if not self.treenodes:
            self.__TreeNodeInsertCFNode__(CFNode)
        else:
            self.__FindBestClusteringNode__(tree_node=self,findcf=CFNode)

    # 按照从根节点中进行遍历找到最优的插入路径
    def __FindBestClusteringNode__(self,tree_node,findcf):
        len_treenode = len(tree_node.treenodes)
        index = 0
        # 从根节点开始遍历，先遍历根节点中簇间距离最短的CFnode，如果其有子节点就继续往下延伸
        for i in range(len_treenode):
            cfnode = tree_node.treenodes[i]
            if i == 0:
                min = tree_node.__Dist_Clustering__(cfnode,findcf)
                index = i
                continue
            temp = tree_node.__Dist_Clustering__(cfnode,findcf)
            if temp < min:
                min = temp
                index = i

        print(index)
        cf_next_treenode = tree_node.treenodes[index]

        # 判断有没有子节点，若有进行递归,有节点进行递归
        if cf_next_treenode.child:
            tree_node.__FindBestClusteringNode__(cf_next_treenode.child,findcf)

        # 没有子节点就进行插入,没有子节点那么这个点一定是叶子节点
        else:

            # 阈值判断
            DivisionJudge = tree_node.__DivisionValue__(tree_node,findcf,tree_node.threshold)

            # 不大于叶子平衡因子，且阈值符合满足
            if len_treenode < tree_node.leaf_balance and DivisionJudge:
                print("Insert")
                tree_node.__TreeNodeInsertCFNode__(findcf)

            # 否则就需要分裂
            else:
                print("Split")
                tree_node.__SplitTwoParts__(tree_node,findcf)


    # 开始分裂，把TreeNode的叶子节点分裂开来变成两个
    def __SplitTwoParts__(self,TN,CF):
        # 建立一个 N*N 的矩阵，其中 i,j 的行列就是 某两个单独簇的距离
        TList = TN.treenodes
        TList.append(CF)
        matri = len(TList)
        KM = np.zeros((matri,matri))
        for i in range(matri):
            for j in range(matri):
                KM[i][j] = self.__Dist_Clustering__(TList[i],TList[j])

        # 初始化两个 treenode 节点 ，和两个 CFnode 节点
        LeftTreeNode = TreeNode(self.branch_balance,self.leaf_balance,self.threshold,TN.is_leaf,self.n_features)
        RightTreeNode = TreeNode(self.branch_balance,self.leaf_balance,self.threshold,TN.is_leaf,self.n_features)

        cf1node = CFNode(np.zeros((1, self.n_features)))
        cf1node.N = LeftTreeNode.N
        cf1node.LS = LeftTreeNode.LS
        cf1node.ES = LeftTreeNode.ES
        cf1node.child = LeftTreeNode
        LeftTreeNode.parent = cf1node

        cf2node = CFNode(np.zeros((1, self.n_features)))
        cf2node.N = RightTreeNode.N
        cf2node.LS = RightTreeNode.LS
        cf2node.ES = RightTreeNode.ES
        cf2node.child = RightTreeNode
        RightTreeNode.parent = cf2node

        # 将节点两两的距离计算出来，然后挑选出最远的两个点作为基本点，将簇划分为两个TreeNode，然后其他点根据与此两点的距离远近，进行填充到两个TreeNode中
        findMaxDist = np.where(KM == np.amax(KM))
        # print(KM)
        # print(findMaxDist)
        A_index = findMaxDist[0][0]
        B_index = findMaxDist[0][1]
        LongDist_List = [A_index,B_index]
        LeftTreeNode.__TreeNodeInsertCFNode__(TList[A_index])
        RightTreeNode.__TreeNodeInsertCFNode__(TList[B_index])

        # 判断其他剩余点与基本点的距离，分配到与自己近的那个点的TreeNode中
        for i in range(matri):
            if i in LongDist_List:
                continue
            if KM[i][A_index] < KM[i][B_index]:
                LeftTreeNode.__TreeNodeInsertCFNode__(TList[i])
            else:
                RightTreeNode.__TreeNodeInsertCFNode__(TList[i])

        '''
            将节点分为两个，然后开始判断原始节点是否是根节点，不是做递归更新，分支分裂等等
        '''

        if TN.parent:
            p_cfnode = TN.parent
            p_treenode = p_cfnode.belong
            p_treenode.__DeleteCFNode__(p_cfnode)

            p_treenode.__TreeNodeInsertCFNode__(cf1node)
            p_len_treenodes = len(p_treenode.treenodes)

            # 小于分支限定大小
            if p_len_treenodes < p_treenode.branch_balance:
                p_treenode.__TreeNodeInsertCFNode__(cf2node)
            else:
                p_treenode.__SplitTwoParts__(p_treenode,cf2node)


        # 如果是根节点，就重置该节点变为新的根节点，树的高度加1
        else:
            TN.__ResetThisNode__()

            TN.__TreeNodeInsertCFNode__(cf1node)
            TN.__TreeNodeInsertCFNode__(cf2node)

    # 重置当前节点，即重置 N ,LS,ES,is_leaf, treenodes
    def __ResetThisNode__(self):
        self.N = 0
        self.LS = np.zeros((1, self.n_features))
        self.ES = 0.0
        self.is_leaf = False
        self.treenodes = []

    # 节点删除
    def __DeleteCFNode__(self,CFNode):
        self.N = self.N - CFNode.N
        self.LS = self.LS - CFNode.LS
        self.ES = self.ES - CFNode.ES
        CFNode.belong = None
        CFNode.child = None
        self.treenodes.remove(CFNode)
        self.__DeleteUpdate__(self,CFNode)

    # 节点删除时候，对应的递归到根节点N,LS,ES
    def __DeleteUpdate__(self,treenode,CFNode):
        if treenode.parent:
            cfnode = treenode.parent
            cfnode.N = treenode.N
            cfnode.LS = treenode.LS
            cfnode.ES = treenode.ES

            belongTree = cfnode.belong
            if belongTree:
                belongTree.N = belongTree.N - CFNode.N
                belongTree.LS = belongTree.LS - CFNode.LS
                belongTree.ES = belongTree.ES - CFNode.ES

                treenode.__DeleteUpdate__(belongTree,CFNode)
        else:
            pass

    # 对于树节点插入CFnode，并更新树节点的聚类特征信息
    def __TreeNodeInsertCFNode__(self,CFNode):
        self.N = self.N + CFNode.N
        self.LS = self.LS + CFNode.LS
        self.ES = self.ES + CFNode.ES
        CFNode.belong = self
        self.treenodes.append(CFNode)
        self.__UpdataPathInformation__(self,CFNode)


    # 更新路径上的所有信息与聚类特征
    def __UpdataPathInformation__(self,treenode,CN):

        if treenode.parent:
            cfnode = treenode.parent
            cfnode.N = treenode.N
            cfnode.LS = treenode.LS
            cfnode.ES = treenode.ES

            belongTree = cfnode.belong
            if belongTree:
                belongTree.N = belongTree.N + CN.N
                belongTree.LS = belongTree.LS + CN.LS
                belongTree.ES = belongTree.ES + CN.ES

                treenode.__UpdataPathInformation__(belongTree,CN)
        else:
            pass


    # 挑选一个统计量来表示作为簇分布的统计量， ρ 【样本点到样本中心点的平均距离，也称为半径】   ， θ 【样本点两两直接的平均距离，也称为直径】
    # 通过返回的统计量，来跟阈值比较，从而判断该节点插入在何处
    def __DivisionValue__(self,C1,C2,threshold):
        N = C1.N + C2.N
        LS = C1.LS + C2.LS
        ES = C1.ES + C2.ES

        LS_Square = (LS**2).sum()
        p_radius = np.sqrt((N*ES - LS_Square)/ N**2)
        d_diameter = np.sqrt((2*N*ES - 2*LS_Square) / (N*(N-1)))

        if p_radius > threshold:
            return False
        else:
            return True

    # 从以下距离中挑选一个方式来当做两个簇之间的距离公式
    def __Dist_Clustering__(self,C1,C2):

        # 形心欧几里得距离
        d0 = np.sqrt((C1.LS/C1.N - C2.LS/C2.N)**2).sum()

        # 形心曼哈顿距离
        d1 = (np.abs(C1.LS/C1.N-C2.LS/C2.N)).sum()

        # 簇联通平均距离
        d2 = np.sqrt(C1.ES/C1.N - 2*(np.dot(C1.LS,C2.LS.T)/(C1.N*C2.N)) + (C2.ES/C2.N)).sum()

        # 全连通平均距离
        d3 = np.sqrt((2*(C1.N+C2.N)*(C1.ES+C2.ES)-2*(((C1.LS+C2.LS)**2).sum()))/((C1.N+C2.N)*(C1.N+C2.N-1)))

        # 散布恶化距离
        d4 = np.sqrt(((C1.N+C2.N)*(C1.ES+C2.ES) - ((C1.LS + C2.LS)**2).sum()) / ((C1.N+C2.N)**2))
        d4 = d4 - (np.sqrt((C1.N*C1.ES - (C1.LS**2).sum())/ C1.N**2) + np.sqrt((C2.N*C2.ES - (C2.LS**2).sum())/ C2.N**2))

        return d4




'''
因为 python 内没有指针这么一项，也没有结构体这么一项，所以考虑可以用 dict 或者 class 来进行操作，这里使用 class 进行操作
'''
class Birch_Class():

    # 获取的初始化参数的值，枝平衡银子β，叶平衡因子λ，空间阈值T，以及是否
    def __init__(self,branch_balance=2,leaf_balance=3,threshold=3,compute_labels=True):
        self.branch_balance = branch_balance
        self.leaf_balance = leaf_balance
        self.threshod = threshold
        self.compute_labels=compute_labels

    # 建立CF特征树
    def CreateCFTree(self,ArrayX,ArrayY=None):
        threshold = self.threshod
        branch_balance = self.branch_balance
        leaf_balance = self.leaf_balance
        if branch_balance <= 1:
            raise ValueError("Branching Balance should be greater than one")
        x_samples,x_features = ArrayX.shape

        # 建立 root 节点，同时也是 叶子节点
        self.root= TreeNode(branch_balance,leaf_balance,threshold,True,x_features)

        # 循环数据
        for sample in ArrayX:
            sample = sample.reshape(1,sample.shape[0])
            cf_node = CFNode(sample)
            self.root.TreeNode_insert_CFNode(cf_node)

        print(self.root)

        self.printss(self.root)

    def printss(self,root):
        lenh = len(root.treenodes)
        for i in range(lenh):
            cfnode = root.treenodes[i]
            if cfnode.child:
                print("has child")
                self.printss(cfnode.child)
            else:
                print(cfnode.LS)



s = np.array([[1,2,3],[1,-2,4],[6,-7,2],[2,-3,7],[1,2,3]])
Bp = Birch_Class()
Bp.CreateCFTree(s)


