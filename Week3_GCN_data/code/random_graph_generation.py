import torch
import torch_geometric
from torch_geometric.utils import to_undirected
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(data, title):
    G = torch_geometric.utils.to_networkx(data)
    colors = ['royalblue', 'darkorange', 'mediumpurple', 'limegreen', 'firebrick']

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    for i, color in enumerate(colors):
        nodes = [node for node in G.nodes() if data.y[node] == i]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=20)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowstyle='-')
    plt.axis('off')
    plt.savefig(title + '.png', dpi = 600)
    plt.show()

if __name__ == '__main__':
    num_nodes = 50  
    num_classes = 5  
    num_edges = 100  

    # 随机生成节点特征
    x = torch.randn(num_nodes, 16)

    # 随机生成节点类别
    y = torch.randint(num_classes, (num_nodes,))

    # 随机生成边的索引
    edge_index = torch.randint(num_nodes, (2, num_edges))
    edge_index = to_undirected(edge_index)

    # 创建PyG的Data对象
    data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)
    visualize_graph(data, 'random_graph1')
    
    # 随机生成训练集、测试集和验证集的掩码
    # train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # train_mask[:int(0.6 * num_nodes)] = 1

    # test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # test_mask[int(0.2 * num_nodes):int(0.8 * num_nodes)] = 1

    # val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # val_mask[int(0.2 * num_nodes):] = 1

