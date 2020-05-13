#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月19日

@author: qilixiang
'''

# try except来捕捉异常
while True:
    try:
        x = int(input("enter a num:"))
        print("input num is " + str(x))
        break
    except NameError:
        print("NameError,agin...")
    except ValueError as error:
        print("ValueError,agin...  Error:\n {}".format(ValueError))  # 打印错误的类型
        print("ValueError,agin...  Error:\n {}".format(error))  # 打印错误的详细信息
    except Exception:
        print("Exception,agin...")
