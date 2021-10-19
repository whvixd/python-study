import torch

'''
自动求梯度
https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter02_prerequisite/2.3_autograd
'''

# Tensor都有一个.grad_fn属性，该属性即创建该Tensor的Function,
# 就是说该Tensor是不是通过某些运算得到的，若是，则grad_fn返回一个与这些运算相关的对象，否则是None。
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)


# 注意x是直接创建的，所以它没有grad_fn, 而y是通过一个加法操作创建的，所以它有一个为<AddBackward>的grad_fn。
#
# 像x这种直接创建的称为叶子节点，叶子节点对应的grad_fn是None。
y = x + 2
print(y)
print(y.grad_fn)

print(x.is_leaf, y.is_leaf)  # x是直接创建的，所以是叶子节点

z=y*y*2
out=z.mean()
print(z,out)
print(z.grad_fn)

x.backward(torch.empty(2,2))
print(x.grad)



a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)

print(out)
out.backward() # 等价于 out.backward(torch.tensor(1.))
print(x.grad)

# r=torch.tensor([[1,2],[1,2]])
# r1=torch.tensor([[3,4],[2,4]])
# e=torch.tensor([[1],[2],[3]])
# print(r.multiply(r))
# print(r*r1)
'''
grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。
'''
out2=x.sum()
out2.backward()
print(x.grad)

out3=x.sum()# 元素之和，结果为标量
print(out3)
x.grad.data.zero_()#梯度清零
out3.backward()
print(x.grad)


x1 = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y1 = 2 * x1
z1 = y1.view(2, 2)
print(z1)
# 现在 z 不是一个标量，所以在调用backward时需要传入一个和z同形的权重向量进行加权求和得到一个标量。
v1 = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z1.backward(v1)
print(x1.grad)
print(x1.data)

# torch.no_grad() 会中断梯度追踪

print(torch.ones(100))

v=torch.ones(1)
v.requires_grad_(True)
v1=v**3
v1=v1**3
v1.backward()
print(v.grad)
v.grad.data.zero_()
