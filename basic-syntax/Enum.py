#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月20日

@author: qilixiang
'''
from enum import Enum, unique


@unique  # 元素是唯一的
class Weekday(Enum):
    Sun = 0  # Sun的value被设定为0
    Mon = 1
    Tue = 2
    Wed = 3
    Thu = 4
    Fri = 5
    Sat = 6


s = Weekday.Sun
print(s)
