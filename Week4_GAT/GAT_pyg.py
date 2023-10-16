import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt

# 加载Cora数据集
dataset = Planetoid(root='../data/Cora', name='Cora', transform = T.NormalizeFeatures())
data = dataset[0]

# 定义GAT模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 定义训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
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
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.tight_layout()
    plt.savefig('GAT_loss.png', dpi = 600)
    plt.show()

    plt.figure()
    plt.plot(range(len(accuracies)), accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.tight_layout()
    plt.savefig('GAT_accuracy.png', dpi = 600)
    plt.show()

if __name__ == '__main__':
    model = GAT(dataset.num_features, hidden_channels=8, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
  
    losses = []
    accuracies = []
    epoches = 200
    for epoch in range(epoches):
        loss = train()
        acc = test()
        losses.append(loss.detach().numpy())
        accuracies.append(acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

    visualize_train(losses, accuracies)
  
    test_acc = test()
    print(f'Final Test Accuracy: {test_acc:.4f}')
