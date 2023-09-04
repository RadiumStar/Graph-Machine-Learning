# WEEK1 GCN
## 论文精读
- Motivation
- Methodology
- Experiments
- 疑问：
## 代码实现
- 参考 [Graph Convolutional Networks in PyTorch](https://github.com/tkipf/pygcn) 和 [GCN-PyTorch](https://github.com/dragen1860/GCN-PyTorch) 代码
- 分析代码框架：todo
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
- [GCN-PyTorch](https://github.com/dragen1860/GCN-PyTorch)