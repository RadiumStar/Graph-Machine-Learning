import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import plotly
import optuna.visualization 
import matplotlib.pyplot as plt

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate = 0.5, layer_size = 1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.hidden_conv = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.layer_size = layer_size
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        for _ in range(self.layer_size - 1):
            x = self.hidden_conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

def objective(trial):
    # 超参数搜索空间
    # layer_size = trial.suggest_int('layer_size', 1, 3)
    layer_size = 1  # number of HIDDEN layers
    epochs = 200    # number of epochs
    hidden_channels = trial.suggest_int('hidden_channels', 16, 128)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.7)

    # 加载数据集
    dataset = Planetoid(root='../../data/' + dataset_name[select], name = dataset_name[select], transform=T.NormalizeFeatures())

    data = dataset[0]
    data = data.to('cuda')

    # 定义模型和优化器
    model = GCN(dataset.num_features, hidden_channels, dataset.num_classes, dropout_rate, layer_size).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    for epoch in range(epochs):
        train(model, optimizer, criterion, data)

    # 评估模型
    acc = evaluate(model, data)

    return acc

if __name__ == '__main__':
    dataset_name = ['Cora', 'CiteSeer', 'PubMed']
    select = 1
    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = 100)

    # 打印最佳超参数和准确率
    best_params = study.best_params
    best_acc = study.best_value
    print('Best Parameters:', best_params)
    print('Best Accuracy:', best_acc)

    contour = optuna.visualization.plot_contour(study)
    plotly.offline.plot(contour, filename = dataset_name[select] + 'contour.html')
    history = optuna.visualization.plot_optimization_history(study)
    plotly.offline.plot(history, filename = dataset_name[select] + 'history.html')
    importance = optuna.visualization.plot_param_importances(study)
    plotly.offline.plot(importance, filename = dataset_name[select] + 'importance.html')
    parallel = optuna.visualization.plot_parallel_coordinate(study, ['hidden_channels', 'lr', 'weight_decay', 'dropout_rate'])
    plotly.offline.plot(parallel, filename = dataset_name[select] + 'parallel.html')
    slices = optuna.visualization.plot_slice(study, ['hidden_channels', 'lr', 'weight_decay', 'dropout_rate'])
    plotly.offline.plot(slices, filename = dataset_name[select] + 'slices.html')


