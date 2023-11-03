import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, temperature = 1, option = 1):
        super(GNNLayer, self).__init__()
        self.W_L = nn.Parameter(torch.randn(in_channels, out_channels))
        self.W_H = nn.Parameter(torch.randn(in_channels, out_channels))
        self.W_I = nn.Parameter(torch.randn(in_channels, out_channels))
        self.tilde_W_L = nn.Parameter(torch.randn(out_channels, 1))
        self.tilde_W_H = nn.Parameter(torch.randn(out_channels, 1))
        self.tilde_W_I = nn.Parameter(torch.randn(out_channels, 1))
        self.W_Mix = nn.Parameter(torch.randn(3, 3))
        self.option = option
        self.T = temperature
        
    def forward(self, X, A):
        # Step0: initialize parameters
        I = torch.eye(A.size(0)).to(X.device)
        tilde_A = A + I
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(tilde_A, dim=1)))
        H_LP = D_inv_sqrt.mm(tilde_A).mm(D_inv_sqrt)
        H_HP = I - H_LP

        # Step1: Feature Extraction for Each Channel
        if self.option == 1: 
            H_L = F.relu(torch.mm(torch.mm(H_LP, X), self.W_L))
            H_H = F.relu(torch.mm(torch.mm(H_HP, X), self.W_H))
            H_I = F.relu(torch.mm(torch.mm(I, X), self.W_I))
        elif self.option == 2:
            H_L = torch.mm(H_LP, F.relu(torch.mm(X, self.W_L)))
            H_H = torch.mm(H_HP, F.relu(torch.mm(X, self.W_H)))
            H_I = torch.mm(I, F.relu(torch.mm(X, self.W_I)))

        # Step2: Row-wise Feature-based Weight Learning
        tilde_alpha_L = torch.sigmoid(torch.mm(H_L, self.tilde_W_L))
        tilde_alpha_H = torch.sigmoid(torch.mm(H_H, self.tilde_W_H))
        tilde_alpha_I = torch.sigmoid(torch.mm(H_I, self.tilde_W_I))
        
        alpha_L = F.softmax((torch.cat((tilde_alpha_L, tilde_alpha_H, tilde_alpha_I), dim=1) / self.T) @ self.W_Mix, dim=1)
        
        # Step3: Node-wise Adaptive Channel Mixing
        H = torch.diag(alpha_L[:, 0]).mm(H_L) + torch.diag(alpha_L[:, 1]).mm(H_H) + torch.diag(alpha_L[:, 2]).mm(H_I)
        H = F.relu(H)
        
        return H

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, temperature = 1, option = 1, dropout_rate = 0.5):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GNNLayer(in_channels, hidden_channels, temperature, option))
            elif i == num_layers - 1:
                self.layers.append(GNNLayer(hidden_channels, out_channels, temperature, option))
            else:
                self.layers.append(GNNLayer(hidden_channels, hidden_channels, temperature, option))
    
    def forward(self, x, a):
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x, a)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    num_nodes = 10  # 节点数量
    in_channels = 16  # 输入特征的维度
    hidden_channels = 32  # 隐藏层特征的维度
    out_channels = 64  # 输出特征的维度
    num_layers = 2  # GNN的层数
    option = 1
    T = 1.0  # 温度参数

    # 生成示例的节点特征矩阵和邻接矩阵
    X = torch.randn(num_nodes, in_channels)
    A = torch.randn(num_nodes, num_nodes)

    model = GNN(in_channels, hidden_channels, out_channels, num_layers, T, option)
    output = model(X, A)

    print(output)
    print("done")
