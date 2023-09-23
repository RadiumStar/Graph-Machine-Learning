import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

dataset = Planetoid(root='data/Cora', name='Cora')

data = dataset[0]

# 定义划分比例
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# 计算划分的节点数量
num_nodes = data.num_nodes
num_train = int(train_ratio * num_nodes)
num_val = int(val_ratio * num_nodes)
num_test = num_nodes - num_train - num_val

# 随机打乱节点索引
perm = torch.randperm(num_nodes)

# 划分节点索引
train_index = perm[:num_train]
val_index = perm[num_train:num_train+num_val]
test_index = perm[num_train+num_val:]

# 创建训练集、验证集和测试集的掩码
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# 将划分的节点索引设置为True
train_mask[train_index] = True
val_mask[val_index] = True
test_mask[test_index] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# 创建GCN模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim = 1)

# 初始化模型和优化器
model = GCN(dataset.num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 定义训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# 定义测试函数
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    return acc

# 训练模型
for epoch in range(200):
    train()
    
acc = test()
print(f'Test Accuracy: {acc:.4f}')
