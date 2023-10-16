import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


dataset = Planetoid(root='../../data/Cora', name='Cora')
data = dataset[0]


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate = 0.6):
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


model = GraphSAGE(in_channels = dataset.num_features, 
                  hidden_channels = 64, 
                  out_channels = dataset.num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    print("loss: ", loss.item())
    loss.backward()
    optimizer.step()


for epoch in range(200):
    train()


model.eval()
out = model(data.x, data.edge_index)
correct = int(out[data.test_mask].max(1)[1].eq(data.y[data.test_mask]).sum())
acc = correct / int(data.test_mask.sum())
print(f'Test Accuracy: {acc:.4f}')
