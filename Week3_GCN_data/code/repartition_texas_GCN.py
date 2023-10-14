import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import WebKB
from torch_geometric.nn import GCNConv

dataset = WebKB(root='data/Texas', name='Texas', transform = T.NormalizeFeatures())
data = dataset[0]
# data.y = F.one_hot(data.y, num_classes=dataset.num_classes).float()

# 定义划分比例
train_ratio = 0.1
val_ratio = 0.45
test_ratio = 0.45

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

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

model = GCN(dataset.num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # pred = F.one_hot(pred, num_classes=dataset.num_classes).float()
    correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test = int(correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test


for epoch in range(1, 201):
    loss = train()

acc = test()
print(f'Test Accuracy: {acc:.4f}')