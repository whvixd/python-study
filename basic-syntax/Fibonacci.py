#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月20日

@author: qilixiang
'''


class Fibonacci(object):
    def __init__(self):
        self.a, self.b = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > 1000:
            raise StopIteration()  # raise引出异常，相当于Java中的throw
        return self.a


for i in Fibonacci():
    print(i)
