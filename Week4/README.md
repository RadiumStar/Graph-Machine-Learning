# Week4 GAT
## GAT in Pytorch
- [GAT_pyg.py](GAT_pyg.py) 是使用Pytorch框架的GAT模型，测试数据集为Cora数据集
- result
    | rounds | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 |
    |:------:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    |accuracy|81.3|82.3|81.6|80.9|83.3|81.8|80.6|82.5|81.7|82.0|
  
    average: **81.8 ± 0.75**

## GCN & GAT with my own `Conv` layers
### GCN
- GCN模型公式
  $$
  H^{(l + 1)} = \sigma(\tilde{D}^{-1/2}(A + I_N)\tilde{D}^{-1/2}H^{(l)}W^{(l)})
  $$
- GCNConv: [GCN_own.py](GCN_own.py)
  ```py
  class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(in_channels, out_channels)))

    def forward(self, x, edge_index):
        adj = self.normalize_adjacency(edge_index)
        x = torch.mm(adj, x)
        x = torch.mm(x, self.weight)
        return x

    def normalize_adjacency(self, edge_index, self_loops = 1):
        num_nodes = torch.max(edge_index) + 1
        adj = torch.zeros((num_nodes, num_nodes))
        
        for i, j in zip(edge_index[0], edge_index[1]):
            adj[i, j] = adj[j, i] = 1

        for _ in range(self_loops):
            adj += torch.eye(num_nodes)

        degree = torch.sum(adj, dim=1)
        degree_sqrt_inv = torch.diag(1 / torch.sqrt(degree))
        adj_normalized = torch.mm(torch.mm(degree_sqrt_inv, adj), degree_sqrt_inv)
        return adj_normalized
  ```
- 以上 `Conv` 层，额外实现了可以设置添加自环个数
- 存在问题：代码运行速度慢

### GAT
todo
