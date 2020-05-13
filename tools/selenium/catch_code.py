#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by wangzhx on 2020/05/13

from selenium import webdriver

chrome_options = webdriver.ChromeOptions()
# 使用headless无界面浏览器模式
# 增加无界面选项
chrome_options.add_argument('--headless')

# 如果不加这个选项，有时定位会出现问题
chrome_options.add_argument('--disable-gpu')

# 启动浏览器，获取网页源代码
browser = webdriver.Chrome(chrome_options=chrome_options)
mainUrl = "https://www.taobao.com/"
browser.get(mainUrl)

print browser.page_source

browser.quit()
