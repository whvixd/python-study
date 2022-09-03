from torch import nn

from d2lzh_pytorch import *

import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def __f1__():
    '''
    线性回归
    y=Xw+b+ϵ
    '''
    num_examples = 1000
    num_inputs = 2
    true_w = [2, -3.4]
    true_b = 4.2

    features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    # 其中噪声项 ϵϵ 服从均值为0、标准差为0.01的正态分布。噪声代表了数据集中无意义的干扰。
    e_ = np.random.normal(0, 0.01, size=labels.size())
    labels += torch.tensor(e_, dtype=torch.float32)
    print(features[0], labels[0])
    # set_figsize()
    # plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)

    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        print(X, y)
        break

    # 需要将随机给定的w,b经过训练后，得到接近真实的w,b
    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)
    print("原始随机w：",w)
    print("原始随机b：",b)

    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    lr = 0.03  # 学习率
    num_epochs = 3  # 迭代周期个数
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
        # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
        # 和y分别是小批量样本的特征和标签
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
            l.backward()  # 小批量的损失对模型参数求梯度
            print("w.grad:",w.grad)# w的梯度就是损失函数对w求导
            # 通过梯度，推导出下一个w,每次循环慢慢接近真实w，l越来越小
            sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
            # 不要忘了梯度清零
            w.grad.data.zero_()
            b.grad.data.zero_()

        # print("训练后w：", w)
        # print("训练后b：", b)
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

    print(true_w, '\n', w)
    print(true_b, '\n', b)


def __f2__():
    # 生成数据集
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

    # 1.读取数据
    import torch.utils.data as Data

    batch_size = 10
    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(features, labels)
    # 随机读取小批量
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    # 2.定义模型
    net = LinearNet(num_inputs)
    print(net)  # 使用print可以打印出网络的结构

    net = nn.Sequential(
        nn.Linear(num_inputs, 1)
        # 此处还可以传入其他层
    )

    for param in net.parameters():
        print(param)

    # 3.初始化模型参数
    from torch.nn import init

    init.normal_(net[0].weight, mean=0, std=0.01)
    init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

    # 定义损失函数
    loss = nn.MSELoss()

    # 定义优化算法
    import torch.optim as optim

    optimizer = optim.SGD(net.parameters(), lr=0.03)
    print(optimizer)

    '''
    optimizer = optim.SGD([
        # 如果对某个参数不指定学习率，就使用最外层的默认学习率
        {'params': net.subnet1.parameters()},  # lr=0.03
        {'params': net.subnet2.parameters(), 'lr': 0.01}
    ], lr=0.03)
    
    # 调整学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1  # 学习率为之前的0.1倍
    '''
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            l.backward()
            optimizer.step()
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        print('epoch %d, loss: %f' % (epoch, l.item()))

        dense = net[0]
        print("----")
        print(true_w, dense.weight)
        print(true_b, dense.bias)

def __f4__():
    dataset=pd.read_csv("/Users/whvixd/Documents/individual/MODIS/dataset/SL/spectral/h2o_data_withMissingS",header=0)

    num_inputs=5
    features_co=dataset.columns[0:5]
    features=torch.tensor(dataset[features_co].values,dtype=torch.float)
    labels=torch.tensor(dataset['B7_lag'].values,dtype=torch.float)

    # 1.读取数据
    import torch.utils.data as Data
    batch_size = 100
    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(features, labels)
    # 随机读取小批量
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    net=Net(num_inputs,30,30,1)
    print(net)

    net = nn.Sequential(
        nn.Linear(num_inputs, 1)
        # 此处还可以传入其他层
    )

    for param in net.parameters():
        print(param)

    # 3.初始化模型参数
    from torch.nn import init

    init.normal_(net[0].weight, mean=0, std=0.01)
    init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

    # 定义损失函数
    loss = nn.MSELoss()

    # 定义优化算法
    import torch.optim as optim

    optimizer = optim.SGD(net.parameters(), lr=0.03)
    print(optimizer)

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            l.backward()
            optimizer.step()
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        print('epoch %d, loss: %f' % (epoch, l.item()))

        dense = net[0]
        print("----")
        print(dense.weight)
        print(dense.bias)



class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden_1,n_hidden_2,n_output): #构造函数
        #构造函数里面的三个参数分别为，输入，中间隐藏层处理，以及输出层
        super(Net,self).__init__() #官方步骤
        self.l1=torch.nn.Linear(n_features, n_hidden_1)
        self.l2=torch.nn.Linear(n_hidden_1,n_hidden_2)
        self.l3=torch.nn.Linear(n_hidden_2,n_output)

    def forward(self,x):  #搭建的第一个前层反馈神经网络  向前传递
        x = F.relu(self.l1(x))# 激活函数，直接调用torch.nn.functional中集成好的Relu
        x=F.relu(self.l2(x))
        x = self.l3(x)  #此行可预测也可以不预测
        return x

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y



if __name__ == '__main__':
    __f4__()