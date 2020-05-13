#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年11月20日

@author: qilixiang
'''
from tkinter import *

import tkinter.simpledialog as dl
import tkinter.messagebox as mb

root = Tk()
w = Label(root, text="Lable Title")
w.pack()

mb.showinfo("welcome", "Hello")

guess = dl.askinteger("Number", "Enter a number")

output = "This is output message"
mb.showinfo("Output", output)
