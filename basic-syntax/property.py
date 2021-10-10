#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月20日

@author: qilixiang
'''

class Student:
    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self,new_name):
        self.__name = new_name
    

xiaohong = Student()
xiaohong.name = "小红"
print(xiaohong.name)