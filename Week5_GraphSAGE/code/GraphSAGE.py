import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

# 加载Cora数据集
dataset = Planetoid(root='../../data/Cora', name='Cora')
data = dataset[0]

# 定义GraphSAGE模型
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

# 创建模型和优化器
model = GraphSAGE(in_channels=dataset.num_features, out_channels=16)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# 训练模型
for epoch in range(10):
    train()

# 测试模型
model.eval()
out = model(data)
correct = int(out[data.test_mask].max(1)[1].eq(data.y[data.test_mask]).sum())
acc = correct / int(data.test_mask.sum())
print(f'Test Accuracy: {acc:.4f}')
