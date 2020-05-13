#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Created by wangzhx on 2020/05/13

"""
id定位：find_element_by_id()
name定位：find_element_by_name()
class定位：find_element_by_class_name()
link定位：find_element_by_link_text()
partial link定位：find_element_by_partial_link_text()
tag定位：find_element_by_tag_name()
xpath定位：find_element_by_xpath()
css定位：find_element_by_css_selector()
"""
import time
from selenium import webdriver

browser = webdriver.Chrome()
browser.get("http://www.baidu.com")
#########百度输入框的定位方式##########
# 通过id方式定位
browser.find_element_by_id("kw").send_keys("selenium")
# 通过name方式定位
browser.find_element_by_name("wd").send_keys("selenium")
# 通过tag name方式定位
browser.find_element_by_tag_name("input").send_keys("selenium")
# 通过class name方式定位
browser.find_element_by_class_name("s_ipt").send_keys("selenium")
# 通过CSS方式定位
browser.find_element_by_css_selector("#kw").send_keys("selenium")
# 通过xpath方式定位
browser.find_element_by_xpath("//input[@id='kw']").send_keys("selenium")
############################################
browser.find_element_by_id("su").click()
time.sleep(3)
browser.quit()
