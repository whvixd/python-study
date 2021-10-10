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

# x=1;y=2
x, y = 1, 2
# 交换值
x, y = y, x
