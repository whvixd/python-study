import unittest
from d2lzh_pytorch import *
import torch
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_softmax(self):
        print(softmax(torch.rand(2, 5)))
        self.assertEqual(True, True)  # add assertion here

    def test_exp(self):
        print(torch.exp(torch.tensor([[2]])))  # e**2
        self.assertEqual(True, True)  # add assertion here

    def test_view(self):
        # 4X3
        x = torch.tensor([[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12]])

        # view-> 6X2
        print(x.view(6, 2))

        # 12/3:int X 3
        print(x.view(-1, 3))
        self.assertEqual(True, True)  # add assertion here

    def test_sum(self):
        x = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])
        print(x.sum())
        self.assertEqual(True, True)  # add assertion here

    def test_gather(self):
        y_hat = torch.tensor([[0.1, 0.3, 0.6],
                              [0.3, 0.2, 0.5]])
        y = torch.LongTensor([0, 2])  # 第一行取index=0的元素：0.1;第二行取index=2的元素：0.5
        print(y_hat.gather(1, y.view(-1, 1)))  # dim=1 行

    def test_ReLU(self):
        # -8.0 ~ 8.0 步长0.1
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        print(x)
        # ReLU(x)=max(x,0) 值域：(0,x)
        y = x.relu()
        print(y)
        xyplot(x, y, 'relu')

        # 求导
        y.sum().backward()
        xyplot(x, x.grad, 'grad of relu')

    def test_sigmoid(self):
        x = torch.arange(-10, 10, 0.1, requires_grad=True)
        # 1/(1+e**-x) 值域：(0,1)
        y = x.sigmoid()
        xyplot(x, y, 'sigmoid')

        y.sum().backward()
        xyplot(x, x.grad, 'grad of sigmoid')

    def test_tanh(self):
        x = torch.arange(-10, 10, 0.1, requires_grad=True)
        # 双曲正切函数:(1-e**-2x)/(1+e**-2x)，值域：(-1,1)
        y = x.tanh()
        # xyplot(x, y, "tanh")
        y.sum().backward()  # y.sum() 为标量
        xyplot(x, y, 'grad of tanh')

    def test_dropout(self):
        X = torch.arange(16).view(2, 8)
        print(X)
        print(dropout(X, 0))

        print(dropout(X, 0.5))
        print(dropout(X, 1.0))

    def test_normal(self):
        print(np.random.normal(0, 0.01, size=(4, 2)))
        print(nn.init.normal_(torch.empty(4, 2), mean=0, std=0.01))

    def test_MSELoss(self):
        loss = nn.MSELoss()
        i_ = torch.tensor([[1, 2], [2, 4]], dtype=torch.float, requires_grad=True)
        o_ = torch.tensor([[2, 3], [4, 2]], dtype=torch.float)
        t_ = loss(i_, o_)  # 先矩阵相减，再每个元素平方除以元素个数
        t_.backward()
        print(t_)

    def test_mul_arg(self):
        def get_mul_arg():
            return 1, 2, 3  # 返回为tuple类型

        def args_(a, b, c):
            print("a:", a)
            print("b:", b)
            print("c:", c)

        args = get_mul_arg()
        print(type(args))
        args_(*args)

    def test_backward(self):
        x = torch.Tensor([4])
        x.requires_grad_(True)
        y = x ** 2
        y.backward()
        # y=x**2 y'=2x ,x=4 y'=8  链式法则求导
        print(x.grad)

    def test_np(self):
        arange_demo = np.arange(0, 1, 0.1, dtype=float)
        print(arange_demo)
        # 创建一位数组
        linspace_demo = np.linspace(1, 100, num=10)
        print(linspace_demo)

        # 等比数列 10**0,10**2
        logspace_demo = np.logspace(0, 2, 20)

        # 索引
        print(logspace_demo[1])

        # 生成2X3的零矩阵
        np.zeros((2, 3), dtype=float)

        # 生成3X5的矩阵，k=0,为对角数为1
        print(np.eye(3, 5, k=0))

        # 生成 3X3矩阵，对角线为 1，3，5
        print("1d np.diag :\n", np.diag([1, 3, 5]))
        # 2维数组，返回对角线的值
        print("2d np.diag:\n", np.diag([[1, 2, 3],
                                        [1, 5, 3],
                                        [1, 2, 7]]))

        # 输入一个100个随机数组
        print(np.random.random(100))

        # 随机生成10X5的随机正态分布矩阵,ravel展平
        randn_demo = np.random.randn(10, 5)
        print(randn_demo.ravel())
        print(randn_demo.flatten(order='F'))

        #         矩阵的创建
        np.mat("123;456;789")

        # pandas处理表格，json相关数据结构

    def test_corr2d(self):
        X = torch.Tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
        K = torch.Tensor([[0, 1],
                          [2, 3]])

        '''
        y11 = x11*k11+x12*k12+x21*k21+x22*k22
        y12 = x12*k11+x13*k12+x22*k21+x23*k22
        y21 = x21*k11+x22*k12+x31*k21+x32*k22
        y22 = x22*k11+x23*k12+x32*k21+x33*k22
        '''
        Y = corr2d(X, K)
        print(Y)

    def test_corr2d_multi_in(self):
        X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                          [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

        print(corr2d_multi_in(X, K))

    def test_stack(self):
        T1 = torch.Tensor([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

        T2 = torch.Tensor([[10, 11, 12],
                           [13, 14, 15],
                           [16, 17, 18]])

        print(torch.stack((T1, T2), dim=0))  # (2,3,3)
        print(torch.stack((T1, T2), dim=1))  # (3,2,3)
        print(torch.stack((T1, T2), dim=2))  # (3,3,2)

    # https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.3_channels
    def test_corr2d_multi_in_out(self):
        X = torch.rand(3, 3, 3)
        K = torch.rand(2, 3, 1, 1)

        Y1 = corr2d_multi_in_out_1x1(X, K)
        Y2 = corr2d_multi_in_out(X, K)

        print((Y1 - Y2).norm().item() < 1e-6)

    def test_norm(self):
        '''
        lp-范数=||X||p = sqrt(p,x1**2+x2**2+...+xn**2) 根号p
        :return: 求l2-范数
        '''
        print(torch.Tensor([[1, 3]]).norm())

    def test_MLP(self):
        from net_demo import MLP
        X = torch.randn(2, 784)

        net = MLP()
        print(net)
        print(net(X))

    def test_sequential(self):
        from net_demo import SequentialClone
        X = torch.randn(2, 784)
        net = SequentialClone(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        print(net)
        print(net(X))

        '''
        ModuleList 类似于数组，顺序传入
        ModuleList仅仅是一个储存各种模块的列表，
        这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），而且没有实现forward功能需要自己实现
        '''
        net = nn.ModuleList([nn.Linear(184, 256), nn.ReLU()])
        net.append(nn.Linear(256, 10))
        print(net)

        '''
        ModuleDict接收一个子模块的字典作为输入, 然后也可以类似字典那样进行添加访问操作
        并没有定义forward函数需要自己定义
        '''
        net = nn.ModuleDict({
            'linear': nn.Linear(784, 256),
            'act': nn.ReLU(),
        })
        net['output'] = nn.Linear(256, 10)  # 添加
        print(net['linear'])  # 访问
        print(net.output)
        print(net)
        # net(torch.zeros(1, 784)) # 会报NotImplementedError

        # named_parameters 访问所以参数
        print(type(net.named_parameters()))
        for name, param in net.named_parameters():
            print(name, param.size())

        for name, param in net.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)
                print(name, param.data)

        # 正态分布初始化参数init.normal_(param, mean=0, std=0.01)
        # 用常数初始化参数init.constant_(param, val=0)

    def test_LeNet(self):
        from net_demo import LeNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = LeNet()
        print(net)

        batch_size = 256
        train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    def test_AlexNet(self):
        from net_demo import AlexNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = AlexNet()
        print(net)
        batch_size = 128
        # 如出现“out of memory”的报错信息，可减小batch_size或resize
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
        # 训练
        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    def test_VGG(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        # 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
        fc_features = 512 * 7 * 7  # c * w * h
        fc_hidden_units = 4096  # 任意

        net = vgg(conv_arch, fc_features, fc_hidden_units)
        X = torch.rand(1, 1, 224, 224)

        # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
        for name, blk in net.named_children():
            X = blk(X)
            print(name, 'output shape: ', X.shape)

        # 因为VGG-11计算上比AlexNet更加复杂，出于测试的目的我们构造一个通道数更小，或者说更窄的网络在Fashion-MNIST数据集上进行训练。
        ratio = 8
        small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio),
                           (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio)]
        net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
        print(net)

        batch_size = 64
        # 如出现“out of memory”的报错信息，可减小batch_size或resize
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    unittest.main()
