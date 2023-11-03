# Week6 Design a GNN
## 1. GNN Structure
- 每层的网络如下：
    Step 1. Feature Extraction for Each Channel:
        $$
        \begin{array}{l}
        \text{Option} 1:  H_{L}^{l}=\operatorname{ReLU}\left(H_{\mathrm{LP}} H^{l-1} W_{L}^{l-1}\right), H_{H}^{l}=\operatorname{ReLU}\left(H_{\mathrm{HP}} H^{l-1} W_{H}^{l-1}\right), H_{I}^{l}=\operatorname{ReLU}\left(I H^{l-1} W_{I}^{l-1}\right) \\
        \text{Option} 2:  H_{L}^{l}=H_{\mathrm{LP}} \operatorname{ReLU}\left(H^{l-1} W_{L}^{l-1}\right), H_{H}^{l}=H_{\mathrm{HP}} \operatorname{ReLU}\left(H^{l-1} W_{H}^{l-1}\right), H_{I}^{l}=I \operatorname{ReLU}\left(H^{l-1} W_{I}^{l-1}\right) \\
        H^{0}=X \in \mathbb{R}^{N \times F_{0}}, W_{L}^{l-1}, W_{H}^{l-1}, W_{I}^{l-1} \in \mathbb{R}^{F_{l-1} \times F_{l}}, l=1, \ldots, L   
        \end{array}
        $$
    Step 2. Row-wise Feature-based Weight Learning
        $$
        \begin{array}{l}
        \tilde{\alpha}_{L}^{l}=\operatorname{Sigmoid}\left(H_{L}^{l} \tilde{W}_{L}^{l}\right), \tilde{\alpha}_{H}^{l}=\operatorname{Sigmoid}\left(H_{H}^{l} \tilde{W}_{H}^{l}\right), \tilde{\alpha}_{I}^{l}=\operatorname{Sigmoid}\left(H_{I}^{l} \tilde{W}_{I}^{l}\right), \tilde{W}_{L}^{l-1}, \tilde{W}_{H}^{l-1}, \tilde{W}_{I}^{l-1} \in \mathbb{R}^{F_{l} \times 1} \\
        {\left[\alpha_{L}^{l}, \alpha_{H}^{l}, \alpha_{I}^{l}\right]=\operatorname{Softmax}\left(\left(\left[\tilde{\alpha}_{L}^{l}, \tilde{\alpha}_{H}^{l}, \tilde{\alpha}_{I}^{l}\right] / T\right) W_{\text {Mix }}^{l}\right) \in \mathbb{R}^{N \times 3}, T \in \mathbb{R} \text { temperature, } W_{\text {Mix }}^{l} \in \mathbb{R}^{3 \times 3} ;}
        \end{array}
        $$

    Step 3. Node-wise Adaptive Channel Mixing:
        $$
        H^{l}=\operatorname{ReLU}\left(\operatorname{diag}\left(\alpha_{L}^{l}\right) H_{L}^{l}+\operatorname{diag}\left(\alpha_{H}^{l}\right) H_{H}^{l}+\operatorname{diag}\left(\alpha_{I}^{l}\right) H_{I}^{l}\right)
        $$

    其中，$H_L^l, H_H^l, H_I^l, H^l$ 表示第 $l$ 层的节点表征
        $$
        H_{LP} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}, \tilde{A} = A + I, H_{HP} = I - H_{LP}
        $$
- 实现2层的GNN网络，中间层的维度不做要求

## 2. Code
1. `GNNlayer`
    ```py
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
    ```

2. `GNN`
    ```py
    class GNN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers, temperature = 1, option = 1, dropout_rate = 0.5, ):
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
    ```

## 3. Result

| accuracy | Cora | CiteSeer |   Texas   |
|:--------:|:----:|:--------:|:---------:|
| option1  |78.13%|  65.90%  |82.70±5.82%|
| option2  |81.40%|  70.50%  |83.54±5.37%|

- `Texas`
   | `mask_id` |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   | avg  |
   |:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|
   |  option1  |89.19 |81.08 |72.97 |86.49 |89.19 |86.49 |75.68 |78.38 |78.38 |89.19 |82.70±5.82|
   |  option2  |86.49 |83.78 |75.68 |91.89 |89.19 |86.49 |81.08 |78.38 |75.68 |81.08 |83.54±5.37|