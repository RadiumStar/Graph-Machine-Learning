import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


# 加载Cora数据集
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

class GATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, concat=True, dropout=0.6):
        super(GATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformations for node features
        self.W = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))

        # Attention mechanisms for each attention head
        self.attn_weights = torch.nn.Parameter(torch.Tensor(2 * out_channels, 1))

        # Initialize weights using Xavier initialization
        torch.nn.init.xavier_uniform_(self.W.data)
        torch.nn.init.xavier_uniform_(self.attn_weights.data)

    def forward(self, x, edge_index):
        h_list = []
        
        for i in range(self.heads):
            # Linear transformation
            h = torch.matmul(x, self.W)  # Use matrix multiplication for the linear transformation
            
            # Self-attention mechanism
            edge_index_i, edge_index_j = edge_index
            edge_index_j = edge_index_j.to(x.device)
            h_i = h[edge_index_i]
            h_j = h[edge_index_j]
            edge_features = torch.cat([h_i, h_j], dim=-1)
            attn_weights = F.leaky_relu(torch.matmul(edge_features, self.attn_weights), negative_slope=0.2)
            attn_weights = F.softmax(attn_weights, dim=0)
            
            # Apply dropout to attention weights
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention to neighbors
            h_j = h_j * attn_weights
            h_head = torch.zeros_like(h)  # Initialize h_head with zeros
            
            # 使用 for 循环遪 edge_index_i 中的索引
            for idx, target_idx in enumerate(edge_index_i):
                h_head[target_idx] += h_j[idx]
                
            h_list.append(h_head)

        if self.concat:
            h_out = torch.cat(h_list, dim=1)
        else:
            h_out = sum(h_list) / len(h_list)
            
        return h_out


# 定义GAT模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat = True, dropout=0.6)
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

if __name__ == '__main__':
    # 初始化模型
    model = GAT(dataset.num_features, hidden_channels=8, num_classes=dataset.num_classes)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # 训练模型
    losses = []
    accuracies = []
    for epoch in range(200):
        loss = train()
        acc = test()
        losses.append(loss.item())
        accuracies.append(acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

    # 打印最终测试结果
    test_acc = test()
    print(f'Final Test Accuracy: {test_acc:.4f}')
