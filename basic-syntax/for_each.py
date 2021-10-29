#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月19日

@author: qilixiang
'''
for i in range(2, 5):  # for(int i=2;i<5;i++)同理
    print(i)
else:
    print("over")

# 遍历List
a_list = [1, 3, 35, 6, 4]
for a in a_list:
    print(a)

b_list=[1 for i in range(10)]
print(b_list)

# 字典的遍历，可以遍历其key和value
b_dic = {'name': 'Tom', 'age': 22, 'score': 99}
b_dic['a'] = 5
for b in b_dic:
    print(b, b_dic[b])

for key in b_dic.items():
    print(key)

for key, value in b_dic.items():
    print(key, value)

print('--test_list--')
# 多线程可能会 出现 IndexError: no such item for Cursor instance，用迭代器
test_list = [1, 2, 3, 4]
for index in range(len(test_list) - 1):
    print(test_list[index])

print('--iter--')
it = iter(test_list)
for index in range(len(test_list) - 1):
    print(next(it))
