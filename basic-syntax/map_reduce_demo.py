'''
map()函数会根据传入的函数对指定的序列做映射。map()函数接收两个参数，
【一个是function函数】，【另一个参数是一个或多个序列。】
map()函数会将传入的函数依次作用到传入序列的每个元素，并把结果作为新的序列返回。
'''
r = map(lambda x: x ** 2, [1, 2, 3, 4])
print(list(r))

r = map(lambda a, b: a + b, [1, 1, 1, 1], [2, 2, 2, 2, 2])
print(list(r))

from functools import reduce

'''
(((1+4)+2)+3)
'''
r = reduce(lambda x, y: x + y, [1, 2, 3], 4)
print(r)
