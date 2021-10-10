#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月19日

@author: qilixiang
'''
# a = input("enter one:")
# b = input("enter two:")
# print("first out is {} second out is {}".format(a, b))
# print("second out is {1} first out is {0}".format(a, b))


print("==========写文件==========")

s = '''面试官好！我叫王志祥，我来自安徽芜湖，
学校是阜阳师范学院，专业是软件工程
'''

# 打开s.txt,并设置为写模式'w' : write
f = open('s.txt', 'w')  # 不写uri，默认在当前目录下
f.write(s)
f.close()

print("==========读文件==========")

r = open('s.txt')  # open默认是'r',所以读文件可以不写'r'
while True:
    line = r.readline()
    if len(line) == 0:
        break
    print(line)
r.close()
