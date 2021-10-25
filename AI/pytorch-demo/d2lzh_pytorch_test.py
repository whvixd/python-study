import unittest
from d2lzh_pytorch import *
import torch


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
        x=torch.tensor([[1,2,3],
                      [4,5,6]])
        print(x.sum())

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
        y.sum().backward() # y.sum() 为标量
        xyplot(x,y,'grad of tanh')



if __name__ == '__main__':
    unittest.main()
