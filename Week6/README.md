# Week6
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

## 2. Train
1. `cora`: 81.40% ![loss](code/GNN_loss.png) ![acc](code/GNN_acc.png)