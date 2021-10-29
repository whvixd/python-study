import torch
from torch import nn

# 通过save函数和load函数可以很方便地读写Tensor。 (使用save可以保存各种对象,包括模型、张量和字典等)
# 通过save函数和load_state_dict函数可以很方便地读写模型的参数。

x = torch.ones(3)
torch.save(x, '../../test/sources/data/x.pt')

x2 = torch.load('../../test/sources/data/x.pt')
print(x2)

y = torch.zeros(4)
torch.save([x, y], '../../test/sources/data/xy.pt')
xy_list = torch.load('../../test/sources/data/xy.pt')
print(xy_list)

torch.save({'x': x, 'y': y}, '../../test/sources/data/xy_dict.pt')
xy = torch.load('../../test/sources/data/xy_dict.pt')
print(xy)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
net.state_dict()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.state_dict()

X = torch.randn(2, 3)
Y = net(X)

PATH = "../../test/sources/data/net.pt"
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y2 == Y)
