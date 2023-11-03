import model
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt

def train(gnn, optimizer, data, device, mask_id):
    gnn.train()
    optimizer.zero_grad()
    out = gnn(data.x, adj)
    loss = F.cross_entropy(out[data.train_mask[:, mask_id]], data.y[data.train_mask[:, mask_id]]).to(device)
    loss.backward()
    optimizer.step()
    return loss.item(), test(gnn, data, device, mask_id)

def test(gnn, data, device, mask_id):
    gnn.eval()
    out = gnn(data.x, adj)
    correct = int(out[data.test_mask[:, mask_id]].max(1)[1].eq(data.y[data.test_mask[:, mask_id]]).sum())
    acc = correct / int(data.test_mask[:, mask_id].sum())
    return acc


def visualize(losses, accuracies, dataname):
    fig1 = plt.figure()
    x = range(len(losses))
    plt.plot(x, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.tight_layout()
    plt.savefig('gallery/GNN_loss_' + dataname + '.png', dpi = 600)
    plt.show()

    fig2 = plt.figure()
    x = range(len(accuracies))
    plt.plot(x, accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.savefig('gallery/GNN_acc_' + dataname + '.png', dpi = 600)
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataname = 'Texas'   
    dataset = WebKB(root='../../data/' + dataname, name = dataname)
    data = dataset[0]
    data = data.to(device)
    adj = to_dense_adj(data.edge_index)[0].to(device)
    mask_id = 0     # select 0 ~ 9

    option = 1
    gnn = model.GNN(in_channels = dataset.num_features,
                    hidden_channels = 16,
                    out_channels = dataset.num_classes,
                    num_layers = 2,
                    temperature = 2.0, 
                    option = option, 
                    dropout_rate = 0.5).to(device)
    
    optimizer = torch.optim.Adam(gnn.parameters(), lr = 0.05, weight_decay = 5e-4)

    epochs = 200
    losses, accuracies = [], []

    for epoch in range(epochs):
        loss, acc = train(gnn, optimizer, data, device, mask_id)
        if epoch % 10 == 0:
            losses.append(loss)
            accuracies.append(acc)
        print("Epoch %d; Loss: %f; Accuracy: %f" % (epoch, loss, acc))

    acc = test(gnn, data, device, mask_id)
    print("Test Accuracy: %.4f" % acc)

    visualize(losses, accuracies, dataname + "_mask_id" + str(mask_id) + "_option" + str(option))
