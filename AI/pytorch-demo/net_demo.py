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

