#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by wangzhx on 2020/05/13

from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
driver = webdriver.Chrome()
driver.get("http://baidu.com")
# 判断title完全等于
title = EC.title_is(u'百度')
print title(driver)

# 判断title包含
title1 = EC.title_contains(u'百度')
print title1(driver)

# 另外一种写法
r1 = EC.title_is(u'百度')(driver)
r2 = EC.title_contains(u'百度')(driver)
print r1
print r2