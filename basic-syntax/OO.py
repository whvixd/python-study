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

class Student:
    # __init__()初始化类，相当于Java中的构造器，self关键字指代该类，相当于Java中的this
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score

    def introduce(self):
        return "name:" + self.name + " age:" + str(self.age) + " score:" + str(self.score)

    def change_age(self, score):
        self.score += score


# 实例化Student对象为tom
tom = Student("Tom", 12, 100)
tom.change_age(-2)
print(tom.introduce())

print("#------------------------------------------------------------------------------ ")


# ===============================================================================
# python与Java相似，也有继承的关系，也体现了多态 
# ===============================================================================

class Animal:
    def __init__(self):
        pass

    def run(self):
        print("running....")


class Dog(Animal):
    def run(self):
        print("There is a dog that is running.")


dahuang = Dog()
animal = Animal
dahuang.run()
animal.run(Animal)

print(type(dahuang) == Dog)
print(isinstance(dahuang, Animal))
