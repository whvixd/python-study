#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月20日

@author: qilixiang
'''


# ===============================================================================
# 支持多继承
# ===============================================================================
class RunnableMixIn(object):
    def run(self):
        print("I can run")


class FlyableMixIn:
    def fly(self):
        print("I can fly")


class Bat(RunnableMixIn, FlyableMixIn):
    pass


black_bat = Bat()
black_bat.fly()
black_bat.run()
