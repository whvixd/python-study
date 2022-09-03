from collections import OrderedDict

import numpy as np
from torch.nn import init

import d2lzh_pytorch as d2l
from d2lzh_pytorch import *


def f1():
    print("load_data_fashion_mnist start")
    # 获取和读取数据
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    print("load_data_fashion_mnist over")

    num_inputs = 784
    num_outputs = 10

    # 定义和初始化模型
    net = LinearNet(num_inputs, num_outputs)

    net = nn.Sequential(
        # FlattenLayer(),
        # nn.Linear(num_inputs, num_outputs)
        OrderedDict([
            ('flatten', FlattenLayer()),
            ('linear', nn.Linear(num_inputs, num_outputs))
        ])
    )

    init.normal_(net.linear.weight, mean=0, std=0.01)
    init.constant_(net.linear.bias, val=0)

    loss = nn.CrossEntropyLoss()

    # 定义优化算法
    # 我们使用学习率为0.1的小批量随机梯度下降作为优化算法。
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练模型
    # 接下来，我们使用上一节中定义的训练函数来训练模型。
    num_epochs = 5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)


def f2():
    def net(X):
        return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs = 784  # 28X28像素
    num_outputs = 10

    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
    b = torch.zeros(num_outputs, dtype=torch.float)

    W.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(X.sum(dim=0, keepdim=True))
    print(X.sum(dim=1, keepdim=True))
    X = torch.rand((2, 5))
    X_prob = softmax(X)
    print(X_prob, X_prob.sum(dim=1))

    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    y = torch.LongTensor([0, 2])
    y_hat.gather(1, y.view(-1, 1))

    print(accuracy(y_hat, y))

    print(evaluate_accuracy(test_iter, net))
    print("===================")
    # 训练模型
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

    X, y = iter(test_iter).next()

    true_labels = get_fashion_mnist_labels(y.numpy())
    pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    show_fashion_mnist(X[0:9], titles[0:9])


if __name__ == '__main__':
    f2()
