import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init()
        self.W_L = nn.Parameter(torch.randn(in_channels, out_channels))
        self.W_H = nn.Parameter(torch.randn(in_channels, out_channels))
        self.W_I = nn.Parameter(torch.randn(in_channels, out_channels))
        self.tilde_W_L = nn.Parameter(torch.randn(out_channels, 1))
        self.tilde_W_H = nn.Parameter(torch.randn(out_channels, 1))
        self.tilde_W_I = nn.Parameter(torch.randn(out_channels, 1))
        self.W_Mix = nn.Parameter(torch.randn(3, 3))
        
    def forward(self, X, A):
        H_L = F.relu(torch.mm(torch.mm(A, X), self.W_L))
        H_H = F.relu(torch.mm(torch.mm(A, X), self.W_H))
        H_I = F.relu(torch.mm(torch.mm(A, X), self.W_I))
        
        tilde_alpha_L = torch.sigmoid(torch.mm(H_L, self.tilde_W_L))
        tilde_alpha_H = torch.sigmoid(torch.mm(H_H, self.tilde_W_H))
        tilde_alpha_I = torch.sigmoid(torch.mm(H_I, self.tilde_W_I))
        
        alpha_L = F.softmax((torch.cat((tilde_alpha_L, tilde_alpha_H, tilde_alpha_I), dim=1) / T) @ self.W_Mix, dim=1)
        
        H = torch.diag(alpha_L[:, 0]).mm(H_L) + torch.diag(alpha_L[:, 1]).mm(H_H) + torch.diag(alpha_L[:, 2]).mm(H_I)
        H = F.relu(H)
        
        return H

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GNN, self).__init()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GNNLayer(in_channels, hidden_channels))
            elif i == num_layers - 1:
                self.layers.append(GNNLayer(hidden_channels, out_channels))
            else:
                self.layers.append(GNNLayer(hidden_channels, hidden_channels))
    
    def forward(self, X, A):
        for layer in self.layers:
            X = layer(X, A)
        return X

# 使用示例
num_nodes = 10  # 节点数量
in_channels = 16  # 输入特征的维度
hidden_channels = 32  # 隐藏层特征的维度
out_channels = 64  # 输出特征的维度
num_layers = 2  # GNN的层数
T = 1.0  # 温度参数

# 生成示例的节点特征矩阵和邻接矩阵
X = torch.randn(num_nodes, in_channels)
A = torch.randn(num_nodes, num_nodes)

model = GNN(in_channels, hidden_channels, out_channels, num_layers)
output = model(X, A)

print(output)
