import torch
from torch import nn


# 自定义多层感知机
class MLP(nn.Module):

    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.hidden(x))
        return self.output(h)
