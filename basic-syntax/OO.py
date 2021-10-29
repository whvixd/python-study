#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月20日
Python支持面向对象编程，还有装饰器
@author: qilixiang
'''

# ===============================================================================
# python设置属性私有化，在属性前面加上两个下划线__,比如__name即为private
# ===============================================================================
from typing import Optional

'''
Python内置类属性
__dict__ : 类的属性（包含一个字典，由类的数据属性组成）
__doc__ :类的文档字符串
__name__: 类名
__module__: 类定义所在的模块（类的全名是'__main__.className'，如果类位于一个导入模块mymod中，那么className.__module__ 等于 mymod）
__bases__ : 类的所有父类构成元素（包含了一个由所有父类组成的元组）

Python 使用了引用计数这一简单技术来跟踪和回收垃圾。
'''


class Student:
    # 全局变量，需要Student.count 访问
    count = 0

    # __init__()初始化类，相当于Java中的构造器，self关键字指代该类，相当于Java中的this
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score
        Student.count += 1

    def __del__(self):
        class_name = self.__class__.__name__
        print(class_name, "销毁")

    def introduce(self):
        return "name:" + self.name + " age:" + str(self.age) + " score:" + str(self.score) + " count:" + str(
            Student.count)

    def change_age(self, score):
        self.score += score


# 实例化Student对象为tom
tom = Student("Tom", 12, 100)
tom.change_age(-2)
print(tom.introduce())

# 销毁对象
del tom

print("#------------------------------------------------------------------------------ ")


# ===============================================================================
# python与Java相似，也有继承的关系，也体现了多态 
# ===============================================================================

class Animal:
    def __init__(self):
        pass

    def run(self):
        print("running....")


class Action:
    def __init__(self):
        pass

    def jump(self):
        print("jump....")


class Dog(Animal, Action):
    def run(self):
        print("There is a dog that is running.")


class Cat(Animal):
    def run(self):
        print("There is a cat that is running.")


dahuang = Dog()
animal = Animal
dahuang.run()
animal.run(Animal)

print(type(dahuang) == Dog)
print(isinstance(dahuang, Animal))
