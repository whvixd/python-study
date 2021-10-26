import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

'''
丢弃法
'''

# 定义模型参数
import torch

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)

W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)

W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2, W3, b3]

# 定义模型
drop_prob1, drop_prob2 = 0.2, 0.5


def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:
        H1 = d2l.dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层（激活之后丢弃）
    H2 = (torch.mm(H1, W2) + b2).relu()
    if is_training:
        H2 = d2l.dropout(H2, drop_prob2)
    return torch.matmul(H2, W3) + b3


num_epochs, lr, batch_size = 4, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 简洁版
net_pytorch=nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1,num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2,num_outputs)
)
for p in params:
    nn.init.normal_(p, mean=0, std=0.01)

# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
optimizer = torch.optim.SGD(net_pytorch.parameters(), lr=0.5)
d2l.train_ch3(net_pytorch, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
