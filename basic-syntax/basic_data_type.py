#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月17日

基本数据类型

@author: qilixiang
'''

print('''111111
    22
333''')

age = 23
name = "王志祥"
print("{0}'age is {1}".format(name, age))
print(name + " is " + str(age))  # 类型不相同的不能打印，需要强转

# * 复制
n2 = name * 2
print(n2)

# 切片
print(name[0: 3])

# 返回有哪些方法
dir(str)
# 查看方法的说明
help(str.count)

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
