from GraphSAGE import GraphSAGE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt


dataset = Planetoid(root='../../data/Cora', name='Cora', transform = T.AddSelfLoops())
data = dataset[0]

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

if __name__ == '__main__':
    accs = []
    for layers in range(1, 9):
        model = GraphSAGE(in_channels = dataset.num_features, 
                          hidden_channels = 16, 
                          out_channels = dataset.num_classes, 
                          hidden_layers = layers)
        optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

        epoches = 200
        losses, accuracies = [], []
        for epoch in range(epoches):
            loss, acc = train(model)
            losses.append(loss)
            accuracies.append(acc)
            # print("Epoch %d; Loss: %f; acc: %f" % (epoch, loss, acc))

        acc = test(model)
        print("hidden_layers: %d; Accuracy: %.4f" % (layers, acc))
        accs.append(acc)

    fig = plt.figure()
    plt.plot(range(1, 9), accs)
    plt.xlabel("hidden layers")
    plt.ylabel("accuracy")
    plt.title("Performance of GraphSAGE in different number of hidden layers")
    plt.grid(True)
    plt.savefig("number_of_hidden_layers_GraphSAGE_performance.png", dpi = 600)
    plt.show()
