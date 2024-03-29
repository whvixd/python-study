#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月18日

@author: qilixiang
'''

print("===========list===========")
num_List = ["a", 1, 2, "3"]
print(num_List)
del num_List[1]  # 删除list的第二个元素
print(num_List)
print('a' in num_List)  # in 判断'a'元素是否在num_List中

# 遍历
for num in num_List:
    print(num)

print(num_List[-1])  # 最后一个元素
print(num_List[0:])
print(num_List[:])
print(num_List[-2:])
print(num_List[:-1])

_list = ['a', 'b', 'c', 'd']

# 扩展
_list.append('e')
# 添加列表
_list.extend(['e', 'f'])
# 指定位子插入
_list.insert(3, 'd')

# 删除指定位子，返回元素
_list.pop(2)

# [1,5) 步长为2（每隔2个取一个元素）
# list_=[1,2,3,4,5]
# list_[0:5,2]
# 1,3,5
print(_list[1:5:2])

# 只删除不返回
del _list[2]

print("===========tuple===========")

a_tuple = (3,)  # 当只有一个元素的时候，后面加上一个逗号，让其识别是元组，如果不加，可能会当它是一个变量

b_tuple = (1, 2, ['a', 'b'])  # 其中的第三个元素是可以改变的，因为改的是数组
# list与tuple的不同：tuple中没有list的append和extend方法，也没有remove和pop方法
# 但是tuple不可变，所以的它的运行速度比List要快，代码更加安全
v = ('a', 'b', 'c')
(x, y, z) = v
print("x:", x)  # tuple支持多赋值

# 因为tuple是不可变的  所以它没有添加，需改，删除元素方法，但是可以删除整个tuple对象

c_tuple = (1, 2, 3, 4, 5)
print(1 in c_tuple)
print(a_tuple + b_tuple)
print(a_tuple * 4)

# 元组的创建
lu1 = (1, 2, 3)
lu2 = 1, 2, 3
lu3 = (1,)

print("===========tuple的截取===========")
print(c_tuple[:])
print(c_tuple[1:2])
print(c_tuple[1:-1])

print("===========dictionary===========")
phone_book = {"张三": 110, "李四": 112}
print(phone_book)
print(phone_book["张三"])
# 可以直接修改字典的值
phone_book['张三'] = 118
# 添加字典的元素
phone_book['王五'] = 187
print(phone_book)
# 删除元素
del phone_book['王五']
# 清空词典
phone_book.clear()
print(phone_book)
# 删除字典这个对象
del phone_book

# 字典中的key是不可重复的
# 可以用元组作为key
messag = {("name",): '小酒'}
print(messag['name'])

# key
messag.keys()
# 值
messag.values()

print("===========set===========")
# 没有重复的key，相当于只有key，没有value的字典
set_ = {1, 2, 3, 4, 4}

s1={'a','b'}
s2={'c','b'}
s3=s1&s2# 交集
s4=s1|s2#并集

# 查看类型
type(phone_book)
