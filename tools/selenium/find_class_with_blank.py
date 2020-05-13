#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by wangzhx on 2020/05/13
import time
from selenium import webdriver

driver = webdriver.Chrome()
driver.get("http://mail.126.com/")

'''
隐性等待，最长等20秒，与time不同的是，time.sleep(20)，强制等待20秒
隐性等待对整个driver的周期都起作用，所以只要设置一次即可，我曾看到有人把隐性等待当成了sleep在用，走哪儿都来一下…
'''


driver.implicitly_wait(20)
print 'login'
# 切换iFrame
driver.switch_to.frame(0)
# 方法一：取单个class属性
driver.find_element_by_class_name("dlemail").send_keys("yoyo")
driver.find_element_by_class_name("dlpwd").send_keys("12333")

# 方法二：定位一组取下标定位（乃下策）
driver.find_elements_by_class_name("j-inputtext")[0].send_keys("yoyo")
driver.find_elements_by_class_name("j-inputtext")[1].send_keys("12333")

# 方法三：css定位
driver.find_element_by_css_selector(".j-inputtext.dlemail").send_keys("yoyo")
driver.find_element_by_css_selector(".j-inputtext.dlpwd").send_keys("123")

# 方法四：取单个class属性也是可以的
driver.find_element_by_css_selector(".dlemail").send_keys("yoyo")
driver.find_element_by_css_selector(".dlpwd").send_keys("123")

# 方法五：直接包含空格的CSS属性定位大法
# driver.find_element_by_css_selector("[class='j-inputtext dlemail']").send_keys("yoyo")

time.sleep(3)
driver.quit()
