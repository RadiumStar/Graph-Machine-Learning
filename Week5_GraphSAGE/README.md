# Week5 GraphSAGE
## Paper Reading: [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)
### Motivation
### Methodology
- 建模过程
  - sample
  - aggregate
- loss
    $$
    J_{\mathcal{G}}\left(\mathbf{z}_{u}\right)=-\log \left(\sigma\left(\mathbf{z}_{u}^{\top} \mathbf{z}_{v}\right)\right)-Q \cdot \mathbb{E}_{v_{n} \sim P_{n}(v)} \log \left(\sigma\left(-\mathbf{z}_{u}^{\top} \mathbf{z}_{v_{n}}\right)\right)
    $$
    $\mathbf{z}_{u}$ 是节点 $u$ 的嵌入向量
    GraphSAGE的loss分为两个部分：前一部分是用于学习正样本的相似性，从而增强正样本节点对节点$u$的相似性；后一部分是负采样损失，用于学习负样本的差异性。在这里，$Q$ 是负样本的数量，$\mathbb{E}_{v_{n} \sim P_{n}(v)}$ 表示从负样本分布 $P_{n}(v)$ 中随机采样 $Q$个负样本节点 $v_{n}$，而 $\mathbf{z}_{v_{n}}$ 是负样本节点 $v_{n}$的嵌入向量，从而增强负样本节点对节点  $u$ 的差异性
### Experiments
### Conclusion

## Transductive & Inductive 
- Transductive: Transductive任务在训练和测试时都可以访问整个图，关注的是对 **已知图** 中的一部分节点进行预测或标签化。在Transductive任务中，模型对整个图的信息是已知的，因此可以根据整个图的上下文来进行预测。
- Inductive: Inductive任务在训练时只能访问一部分节点和它们的邻居，关注的是在模型从训练数据中学到的信息的基础上，对以前未见过的节点进行预测，在测试时要处理整个图，包括从未见过的节点。因此要求模型必须泛化到未见过的节点
- Difference between Transductive and Inductive: 所预测的样本在模型训练的时候已经使用或者访问过了，为transductive；否则为inductive

## Conference
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)

