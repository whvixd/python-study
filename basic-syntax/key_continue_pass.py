#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月19日

@author: qilixiang
'''
a_tuple = [0, 1, 2]
for i in a_tuple:
    if not i:  # python中没有boolean类型，True = 1, False = 0
        continue  # 调出本次循环
    print(i)

for j in a_tuple:
    if not j:
        pass  # 忽略当前的判断，不会结束循环，相当于一个占位符 什么也不做
    print(j)
