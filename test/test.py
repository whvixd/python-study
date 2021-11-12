import unittest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

w = torch.tensor([4], dtype=torch.float)
b = torch.tensor([2], dtype=torch.float)
print("w:", w)
print("b:", b)
w.requires_grad_(True)
b.requires_grad_(True)
y = w ** 2 + b ** 2
print("y:", y)
y.backward()  # 求梯度
print("w':", w.grad)  # 对w求偏导 ay/aw=2w+0
print("b':", b.grad)  # 对b求偏导 ay/aw=0+2b

w.data -= 0.02 * w.grad

print('w:', w.data)
