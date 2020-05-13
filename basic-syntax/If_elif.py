#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月19日

@author: qilixiang
'''
num = 1

while (True):
    input_num = int(input("输入数字："))
    if (num == input_num):
        print("Yes")
        break

    elif (input_num > num):
        print("N0！大了")

    elif (input_num < num):
        print("No！小了")
