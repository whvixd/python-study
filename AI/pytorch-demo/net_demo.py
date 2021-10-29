import torch
from torch import nn
from collections import OrderedDict


# 自定义多层感知机
class MLP(nn.Module):

    def __init__(self, **kwargs):
        # 参数，如模型参数的访问、初始化和共享
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        self.act = nn.ReLU()  # 激活函数
        self.output = nn.Linear(256, 10)  # 输出层

    # 正向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.hidden(x))
        return self.output(h)


X = torch.randn(2, 784)


# net = MLP()
# print(net)
# print(net(X))


class SequentialClone(nn.Module):

    def __init__(self, *args):
        super(SequentialClone, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input


net = SequentialClone(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(net)
print(net(X))
