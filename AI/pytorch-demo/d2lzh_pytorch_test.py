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


if __name__ == '__main__':
    unittest.main()
