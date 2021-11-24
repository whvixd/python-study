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
b_list.pop()

# 字典的遍历，可以遍历其key和value
b_dic = {'name': 'Tom', 'age': 22, 'score': 99}
b_dic['a'] = 5
for b in b_dic:
    print(b, b_dic[b])

for key in b_dic.items():
    print(key)

for key, value in b_dic.items():
    print(key, value)

# dict函数创建
dic_=dict([('a',1),('b',2)])

print('--test_list--')
# 多线程可能会 出现 IndexError: no such item for Cursor instance，用迭代器
test_list = [1, 2, 3, 4]
for index in range(len(test_list) - 1):
    print(test_list[index])

print('--iter--')
it = iter(test_list)
for index in range(len(test_list) - 1):
    print(next(it))

set_={1,1,2,3,2}
set_t=set(("长江", "黄河", "湘江"))

set_co={i for i in [1,2,3,4,-1] if i>1}

colors = {'red','blue','pink'}
sizes = {36,37,38,39}
result = {c + str(s) for c in colors for s in sizes}

# zip函数转成字典
dict_2=zip(sizes,colors)