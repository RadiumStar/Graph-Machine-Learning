import model
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt

def train(gnn, optimizer, data, device):
    gnn.train()
    optimizer.zero_grad()
    out = gnn(data.x, adj)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).to(device)
    loss.backward()
    optimizer.step()
    return loss.item(), test(gnn, data, device)

def test(gnn, data, device):
    gnn.eval()
    out = gnn(data.x, adj)
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
    plt.savefig('GNN_loss.png', dpi = 600)
    plt.show()

    fig2 = plt.figure()
    x = range(len(accuracies))
    plt.plot(x, accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.savefig('GNN_acc.png', dpi = 600)
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = Planetoid(root='../../data/Cora', name='Cora')
    data = dataset[0].to(device)
    adj = to_dense_adj(data.edge_index)[0].to(device)

    gnn = model.GNN(in_channels = dataset.num_features,
                    hidden_channels = 16,
                    out_channels = dataset.num_classes,
                    num_layers = 2,
                    temperature = 2.0, 
                    option = 2, 
                    dropout_rate = 0.6).to(device)
    
    optimizer = torch.optim.Adam(gnn.parameters(), lr = 0.05, weight_decay = 5e-4)

    # print(data.x.size(), data.edge_index.size())
    epochs = 200
    losses, accuracies = [], []

    for epoch in range(epochs):
        loss, acc = train(gnn, optimizer, data, device)
        losses.append(loss)
        accuracies.append(acc)
        print("Epoch %d; Loss: %f; Accuracy: %f" % (epoch, loss, acc))

    acc = test(gnn, data, device)
    print("Test Accuracy: %.4f" % acc)

    visualize(losses, accuracies)