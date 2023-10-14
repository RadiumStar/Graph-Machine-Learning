import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt

# 加载Cora数据集
dataset = Planetoid(root='data/Cora', name='Cora', transform = T.NormalizeFeatures())
data = dataset[0]

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


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.6):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        return F.softmax(x, dim=1)

    
# 定义训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# 定义测试函数
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(acc.sum()) / int(data.test_mask.sum())
    return acc

# 可视化训练过程
def visualize_train(losses, accuracies):
    # 展示训练损失的收敛曲线
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.tight_layout()
    plt.savefig('GCN_loss.png', dpi = 600)
    plt.show()

    # 展示验证准确率的收敛曲线
    plt.figure()
    plt.plot(range(len(accuracies)), accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.tight_layout()
    plt.savefig('GCN_accuracy.png', dpi = 600)
    plt.show()

if __name__ == '__main__':
    # 初始化模型
    model = GCN(in_channels = dataset.num_features, hidden_channels = 16, out_channels = dataset.num_classes)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # 训练模型
    losses = []
    accuracies = []
    for epoch in range(300):
        loss = train()
        acc = test()
        losses.append(loss.detach().numpy())
        accuracies.append(acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')


    visualize_train(losses, accuracies)
    # 打印最终测试结果
    test_acc = test()
    print(f'Final Test Accuracy: {test_acc:.4f}')