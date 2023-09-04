# WEEK1 GCN
## 论文精读
### Motivation
- 把卷积放到Graph这种非欧结构上，充分利用图中邻居节点的性质和图的拓扑结构
- 降低之前的谱图/传播方法的模型的复杂度，并达到好的分类效果
### Methodology
- 核心公式
  - 传播公式
    $$
    H^{(l + 1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})
    $$
    其中，$\tilde{A} = A + I_N$（为了纳入节点自身信息），$\tilde{D}_{ii} = \sum_{j}\tilde{A}_{ij}$，$W$是训练的权重矩阵，$H^{(0)} = X$特征矩阵
  - loss
    $$
    \mathcal{L} = -\sum_{l\in \mathcal{Y_L}}\sum_{f = 1}^{F}Y_{lf}\ln Z_{lf}
    $$
    所有labeled nodes的交叉熵损失，$\mathcal{Y}$是labeled nodes的下标集合，$Z$是最终输出
- 建模
  - 两层GCN，其中$\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$
    $$
    Z = f(X, A) = \text{softmax}\left(\hat{A}\text{ReLU}(\hat{A}XW^{(0)})\right)
    $$
- 理论说明：其实文章思想很简单，就是 **局部一阶近似** 来聚合邻居节点信息，但是需要谱图，傅里叶等一系列说明，才能顺其自然将卷积搬进图这种非欧结构的数据 [todo]()
### Experiment
- 验证GCN在不同模型方法下的半监督节点分类准确率表现好；方法：比较不同模型（label propagation, semi-supervised embedding, manifold regularization, skip-gram based graph embeddings）在不同数据集上的分类准确率
- 验证GCN在都是传播模型下的表现最好，说明他们的Renormalization trick表现最好
## 代码实现
- 参考 [Graph Convolutional Networks in PyTorch](https://github.com/tkipf/pygcn) 和 [GCN-PyTorch](https://github.com/dragen1860/GCN-PyTorch) 代码
- 分析代码框架：[todo]()
- 测试结果
  - Cora数据集 准确率

  |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   | 10 | 平均准确率 |
  |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:--:|:---------:|
  |0.8240|0.8300|0.8270|0.8380|0.8320|0.8230|0.8310|0.8260|0.8380|0.8250|**0.8294**|
  - Citeseer数据集 准确率

  |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   | 10 | 平均准确率 |
  |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:--:|:---------:|
  |0.7040|0.7070|0.6910|0.7140|0.7070|0.7000|0.7120|0.7180|0.7020|0.7140|**0.7069**|
  - Pumbed数据集 准确率

  |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   | 10 | 平均准确率 |
  |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:--:|:---------:|
  |0.7880|0.7820|0.7920|0.7840|0.7930|0.7970|0.7880|0.7980|0.7810|0.7930|**0.7896**|

## 参考
- [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf)
- [Graph Convolutional Networks in PyTorch](https://github.com/tkipf/pygcn)
- [图卷积网络 GCN Graph Convolutional Network（谱域GCN）的理解和详细推导](https://blog.csdn.net/yyl424525/article/details/100058264#1__2)
