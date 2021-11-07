import unittest
from d2lzh_pytorch import *
import torch
import numpy as np
import random
import zipfile


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

    def test_zip(self):
        with zipfile.ZipFile('../../test/sources/data/jaychou_lyrics.txt.zip') as zin:
            with zin.open('jaychou_lyrics.txt') as f:
                corpus_chars = f.read().decode('utf-8')
        corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
        corpus_chars = corpus_chars[0:10000]
        idx_to_char = list(set(corpus_chars))
        char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
        vocab_size = len(char_to_idx)
        vocab_size  # 1027
        corpus_indices = [char_to_idx[char] for char in corpus_chars]
        sample = corpus_indices[:20]
        print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
        print('indices:', sample)


if __name__ == '__main__':
    unittest.main()
