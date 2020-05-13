#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by wangzhx on 2020/05/13

from selenium import webdriver

option = webdriver.ChromeOptions()
option.add_argument('--user-data-dir=/Users/whvixd/Library/Application Support/Google/Chrome/Default')  # 设置成用户自己的数据目录
driver = webdriver.Chrome(chrome_options=option)
