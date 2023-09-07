import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F

import  numpy as np
from    data import load_data, preprocess_features, preprocess_adj
from    model import GCN
from    config import  args


def calculate_loss(out, label, mask):
    # 其中 mask 是一个索引向量，值为1表示该位置的标签在训练数据中是给定的
    # 那么计算损失的时候，loss 乘以的 mask 等于 loss 在未带标签的地方都乘以0没有了，而在带标签的地方损失变成了mask倍
    # 即只对带标签的样本计算损失
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss

def calculate_accuracy(out, label, mask):
    """ accuracy with masking """
    pred = out.argmax(dim = 1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

# load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
print('adj:', adj.shape)
print('features:', features.shape)
print('y:', y_train.shape, y_val.shape, y_test.shape)
print('mask:', train_mask.shape, val_mask.shape, test_mask.shape)

# D^-1@X
features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
supports = preprocess_adj(adj)

device = torch.device('cuda')
train_label = torch.from_numpy(y_train).long().to(device)
num_classes = train_label.shape[1]
train_label = train_label.argmax(dim=1)
train_mask = torch.from_numpy(train_mask.astype(np.int64)).to(device)
val_label = torch.from_numpy(y_val).long().to(device)
val_label = val_label.argmax(dim=1)
val_mask = torch.from_numpy(val_mask.astype(np.int64)).to(device)
test_label = torch.from_numpy(y_test).long().to(device)
test_label = test_label.argmax(dim=1)
test_mask = torch.from_numpy(test_mask.astype(np.int64)).to(device)

i = torch.from_numpy(features[0]).long().to(device)
v = torch.from_numpy(features[1]).to(device)
feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)

i = torch.from_numpy(supports[0]).long().to(device)
v = torch.from_numpy(supports[1]).to(device)
support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

print('x :', feature)
print('sp:', support)
num_features_nonzero = feature._nnz()
feat_dim = feature.shape[1]

net = GCN(feat_dim, num_classes, num_features_nonzero)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

net.train()
for epoch in range(args.epochs):
    out = net((feature, support))
    out = out[0]
    loss = calculate_loss(out, train_label, train_mask)
    loss += args.weight_decay * net.l2_loss()

    acc = calculate_accuracy(out, train_label, train_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("epoch: {}; loss: {:.6f}; acc: {:.6f}".format(epoch, loss.item(), acc.item()))

net.eval()

out = net((feature, support))
out = out[0]
acc = calculate_accuracy(out, test_label, test_mask)
print('test accuracy:', acc.item())