#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by wangzhx on 2020/05/13

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import time, pandas as pd

div_list = []
test_list = []
# 声明配置好的浏览器对象
browser = webdriver.Chrome()
try:
    # 打开网站，搜索
    browser.get('https://www.zhipin.com/job_detail/?ka=header-job')
    # 设置延时
    time.sleep(1)
    input = browser.find_element_by_class_name('ipt-search')
    input.send_keys('python爬虫')
    time.sleep(0.5)
    input.send_keys(Keys.ENTER)

    # 获取职位信息
    divdata = browser.find_element_by_class_name('job-list')
    data = divdata.text
    # 数据筛选
    div_list.append(data.replace("\n", " "))
    print(div_list)
    test = div_list[0].replace("Python", ",Python")
    test = test.split(",")
    print(test)
    for index, i in enumerate(test):
        if len(i) == 0:
            del test[index]
        j = test[index].replace(" ", ",")
        k = j.split(",")
        del k[-1]
        test_list.append(k)
        # test[index].split(" ")
    print(test_list)
    # [['Python', '8k-12k', '上海', '浦东新区', '张江3-5年本科', 'ZTE', '通信/网络设备已上市10000人以上', '曹先生招聘者']]
finally:
    browser.close()
