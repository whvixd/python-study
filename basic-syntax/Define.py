#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2017年11月19日

@author: qilixiang
'''


# from builtins import str


def say_hi():
    print("hello")


say_hi()


def sum(a, b):
    c = a + b
    print(c)


sum(1, 2)


def add(a, b):
    return a + b


sum = add(3, 4)
print(sum)

x = 2


def print_x(y):
    global x  # 声明x为全局变量
    x = y
    print(x)


print_x(4)
print(x)


# 函数中形参可以默认初始值
def foo(str, time=2):
    str2 = str * time
    print(str2)


foo("repeat ", 3)


def foo2():
    print("foo2")


foo2()

print("=====================")


# 函数指定传参：关键字参数
def foo3(a, b=3, c=4):
    print(a, b, c)


foo3(b=5, a=3)

print("=====================")


# VarArgs参数  :  *~ 以元组的方式存储  **~ 以字典的方式存储
def foo4(x=1, *nums, **word):
    print("x：" + str(x))
    print("nums：" + str(nums))
    print("word：" + str(word))


foo4(2, 2, 1, second_word='o', first_word='w', last_word='d')
