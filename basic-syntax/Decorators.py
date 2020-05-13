#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2017年11月20日
装饰器，也是Python特点之一，装饰器的作用就是为已经存在的函数或对象添加额外的功能。
@author: qilixiang
'''


def add_func(func):
    def wrapper():
        return func("hello") + " and hello"

    return wrapper


@add_func  # add_func函数的返回值作为参数，再后面执行函数时，可以不需传参
def say_hello(hello):
    return hello


print(say_hello())


# ------------------------------------------------------------------------------

def print_1():
    print("this is print_1")


# 可以声明一个内切的方法，也可以返回一个方法
def print_2():
    print_1()
    print(" and this is print_2")

    def wrapper():
        print("this is print_2's inner method wrapper")

    wrapper()


print_2()
