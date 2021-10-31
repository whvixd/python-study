import torch
from torch import nn
import d2lzh_pytorch as d2l


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return d2l.corr2d(x, self.weight) + self.bias


# 0为黑色，1为白色
x = torch.ones(6, 8)
x[:, 2:6] = 0

# 白到黑的边缘用1表示，黑到白的边缘用-1表示
k = torch.Tensor([[1, -1]])
y = d2l.corr2d(x, k)

# 现在利用输入x和输出y来学习构造我们的核数组k

# 构造一个核数组形状是(1, 2)的二维卷积层


conv2d = Conv2D((1, 2))
step = 20
lr = 0.01
for i in range(step):
    y_hat = conv2d(x)

    # 平方差损失函数
    l = ((y_hat - y) ** 2).sum()

    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清零
    conv2d.weight.grad.zero_()
    conv2d.bias.grad.fill_(0)

    print("i:%d,loss:%.3f" % (i + 1, l.item()))

print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)