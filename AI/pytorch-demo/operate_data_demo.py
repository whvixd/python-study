import torch

'''
https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter02_prerequisite/2.2_tensor

"tensor"这个单词一般可译作“张量”，张量可以看作是一个多维数组。标量可以看作是0维张量，向量可以看作1维张量，矩阵可以看作是二维张量。
'''
# 然后我们创建一个5x3的未初始化的Tensor：
x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

# 直接创建
x = torch.tensor([1, 2])
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

print(x.size())
print(x.shape)

# 矩阵加法
y = torch.rand(5, 3)
print(x + y)
z = torch.add(x, y)

y.add_(x)
print(y)

'''
索引

'''
# 索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。
j = x[0, :]
j += 1
print(j)
print(x[0, :])

print(x)
print(torch.index_select(x, 0, torch.tensor([0, 3])))  # 0,3 第0行和第3行
print(torch.index_select(x, 1, torch.tensor([0, 2])))  # 0,3 第0列和第2列

'''
改变形状
'''

y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来
print(y)
print(z)

print(torch.ones(1).item())  # item 转为Python number

'''
广播机制
由于x和y分别是1行2列和3行1列的矩阵，如果要计算x + y，那么x中第一行的2个元素被广播（复制）到了第二行和第三行，
而y中第一列的3个元素被广播（复制）到了第二列。如此，就可以对2个3行2列的矩阵按元素相加。
'''
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x  # 写的y的原内存中
print(id(y) == id_before)  # id函数返回内存地址

# tensor 与numpy互转
a = torch.tensor([1, 2, 3])
b = a.numpy()
print(a)
print(b)
import numpy as np

print(torch.from_numpy(np.ones(4)))

# 用方法to()可以将Tensor在CPU和GPU（需要硬件支持）之间相互移动。
if torch.cuda.is_available():
    print("GPU")
    device = torch.device("cuda")  # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)  # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # to()还可以同时更改数据类型
