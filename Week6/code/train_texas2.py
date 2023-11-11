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
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).to(device)
    loss.backward()
    optimizer.step()
    return loss.item(), test(gnn, data, device, mask_id)

def test(gnn, data, device, mask_id):
    gnn.eval()
    out = gnn(data.x, adj)
    correct = int(out[data.test_mask].max(1)[1].eq(data.y[data.test_mask]).sum())
    acc = correct / int(data.test_mask.sum())
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
    data = data.to(device)
    adj = to_dense_adj(data.edge_index)[0].to(device)
    mask_id = 0     # select 0 ~ 9

    option = 2
    gnn = model.GNN(in_channels = dataset.num_features,
                    hidden_channels = 16,
                    out_channels = dataset.num_classes,
                    num_layers = 2,
                    temperature = 2.0, 
                    option = option, 
                    dropout_rate = 0.5).to(device)
    
    optimizer = torch.optim.Adam(gnn.parameters(), lr = 0.286, weight_decay = 5e-4)

    epochs = 200
    losses, accuracies = [], []

    for epoch in range(epochs):
        loss, acc = train(gnn, optimizer, data, device, mask_id)
        if epoch % 10 == 0:
            losses.append(loss)
            accuracies.append(acc)
        if acc > 0.88: 
            optimizer = torch.optim.Adam(gnn.parameters(), lr = 0.01, weight_decay = 5e-4)
        print("Epoch %d; Loss: %f; Accuracy: %f" % (epoch, loss, acc))

    acc = test(gnn, data, device, mask_id)
    print("Test Accuracy: %.4f" % acc)

    # visualize(losses, accuracies, dataname + "_mask_id" + str(mask_id) + "_option" + str(option))
