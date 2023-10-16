import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
import matplotlib.pyplot as plt


dataset = Planetoid(root='../../data/Cora', name='Cora', transform = T.AddSelfLoops())
data = dataset[0]


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate = 0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(model):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item(), test(model)


def test(model):
    model.eval()
    out = model(data.x, data.edge_index) 
    correct = int(out[data.test_mask].max(1)[1].eq(data.y[data.test_mask]).sum())
    acc = correct / int(data.test_mask.sum())
    return acc


def visualize(losses, accuracies):
    fig1 = plt.figure()
    x = range(len(losses))
    plt.plot(x, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.tight_layout()
    plt.savefig('GraphSAGE_loss.png', dpi = 600)
    plt.show()

    fig2 = plt.figure()
    x = range(len(accuracies))
    plt.plot(x, accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.savefig('GraphSAGE_acc.png', dpi = 600)
    plt.show()


if __name__ == '__main__':
    model = GraphSAGE(in_channels = dataset.num_features, 
                    hidden_channels = 16, 
                    out_channels = dataset.num_classes)
    optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

    epoches = 200
    losses, accuracies = [], []
    for epoch in range(epoches):
        loss, acc = train(model)
        losses.append(loss)
        accuracies.append(acc)
        print("Epoch %d; Loss: %f; acc: %f" % (epoch, loss, acc))

    acc = test(model)
    print("Accuracy: %.4f" % acc)

    # visualize(losses, accuracies)
