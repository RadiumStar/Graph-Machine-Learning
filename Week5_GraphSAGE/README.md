# Week5 GraphSAGE
## Paper Reading: [Inductive Representation Learning on Large Graphs](https://github.com/RadiumStar/GraphML_Training/blob/main/Week5_GraphSAGE/Inductive%20Representation%20Learning%20on%20Large%20Graphs.pdf)
### Motivation
### Methodology
### Experiments
### Conclusion

## Transductive & Inductive 
- Transductive: Transductive任务在训练和测试时都可以访问整个图，关注的是对 **已知图** 中的一部分节点进行预测或标签化。在Transductive任务中，模型对整个图的信息是已知的，因此可以根据整个图的上下文来进行预测。
- Inductive: Inductive任务在训练时只能访问一部分节点和它们的邻居，关注的是在模型从训练数据中学到的信息的基础上，对以前未见过的节点进行预测，在测试时要处理整个图，包括从未见过的节点。因此要求模型必须泛化到未见过的节点
- Difference between Transductive and Inductive: 所预测的样本在模型训练的时候已经使用或者访问过了，为transductive；否则为inductive

## 