import torch
import numpy as np
import d2lzh_pytorch as d2l
from d2lzh_pytorch import *
from torch.nn import init


def __manual():
    # 获取和读取数据
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 定义模型参数
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
    b1 = torch.zeros(num_hiddens, dtype=torch.float)
    W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
    b2 = torch.zeros(num_outputs, dtype=torch.float)

    params = [W1, b1, W2, b2]
    for param in params:
        param.requires_grad_(requires_grad=True)

    # 定义模型
    def net(X):
        X = X.view(-1, num_inputs)
        H = relu(torch.matmul(X, W1) + b1)
        return torch.matmul(H, W2) + b2

    # 交叉熵损失函数
    loss = torch.nn.CrossEntropyLoss()

    # 训练模型
    num_epochs, lr = 5, 100.0
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


def __simplified():
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
    )

    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

    # 训练模型
    num_epochs = 5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)


if __name__ == '__main__':
    # __manual()
    __simplified()