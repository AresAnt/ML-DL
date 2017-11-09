PCA
===

标签（空格分隔）： Simple_PCA 

---

PCA：
----

在多元统计分析中，主成分分析（英语：Principal components analysis，PCA）是一种分析、简化数据集的技术。主成分分析经常用于减少数据集的维数，同时保持数据集中的对方差贡献最大的特征。这是通过保留低阶主成分，忽略高阶主成分做到的。这样低阶成分往往能够保留住数据的最重要方面。但是，这也不是一定的，要视具体应用而定。由于主成分分析依赖所给数据，所以数据的准确性对分析结果影响很大。

Simple PCA:
-----------

 暂且我将它定义为简单的PCA，直接通过求取最大的前N项特征值，来得到PCA的解。即W
 详细的简单的数学原理可以参照这个超级链接：[此处输入链接的描述][1]

算法描述过程如下：
输入：样本集 D = {x1,x2,x3,...,xn}
      低维空间维数d'.
过程：
    - 对所有样本进行中心化（归零化与标准化
    - 计算样本的协方差矩阵 XXT 
    - 对协方差矩阵XXT做特征值分解 
    - 取最大的d'个特征值所对应的特征向量（w1，w2，....，wd'）
输出：  矩阵W

中心化公式： ( μ 表示平均值 , σ 表示标准差 )
$ x' = \frac{x - μ}{σ} $

流程图：
```flow
st=>start: 传入数据
op1=>operation: 数据不缺失
op11=>operation: 填充缺失值，以上下邻近数据进行填充
op2=>operation: 数据没有非数值型数据
op22=>operation: 用LabelEncoder进行数据转化
cond1=>condition: Yes or No?
cond2=>condition: Yes or No?
op3=>operation: 将非数值型数据转化成数值型数据
op4=>operation: 中心化以及PCA降维
e=>end

st->op1->cond1->op2->cond2->op3->op4->e
cond1(yes)->op2
cond1(no)->op11->op2
cond2(yes)->op3
cond2(no)->op22->op3
```

  [1]: http://www.360doc.com/content/13/1124/02/9482_331688889.shtml
