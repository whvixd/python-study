#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by wangzhx on 2020/05/13
import sys
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

reload(sys)
sys.setdefaultencoding("utf-8")

browser = webdriver.Chrome()
browser.get("https://www.zhaopin.com")


def login(account, password):
    # 点击 "知道了"
    browser.find_element_by_xpath("//div[@class='risk-warning__content']//button").click()

    # me-login
    login_class = browser.find_element_by_xpath("//div[@class='me-login']")
    # 未登陆
    if login_class is not None:
        print account + " 正在登陆\n"
        login_class.find_element_by_xpath(
            "//div[@class='zp-passport-widget-by-username__input-box zp-passport-widget-by-username__username-box']//input").send_keys(
            account)
        login_class.find_element_by_xpath(
            "//div[@class='zp-passport-widget-by-username__input-box zp-passport-widget-by-username__password-box']//input").send_keys(
            password)
        # zp-passport-widget-plugin
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[@class='zp-passport-widget-plugin']//button")))
        # time.sleep(100)
        login_class.find_element_by_xpath("//button[@class='zp-passport-widget-by-username__submit']").click()
        # todo 若出现，滑动图片，需要手动处理
        if browser.find_element_by_xpath("//div[@class='geetest_panel_next']") is not None:
            print "滑动图片，需要手动处理"
            # 停留5秒
            time.sleep(5)
        print account + " 登陆结束\n"


def find():
    # 跳转到登陆页面
    browser.get("https://i.zhaopin.com/")
    confirmed = browser.find_element_by_xpath("//div[@class='privacy-protocol-update__confirm']")

    if confirmed is not None:
        print "登陆成功"
        # 同意
        confirmed.find_element_by_xpath("//button[text()='同意']").click()

    find_input = browser.find_element_by_xpath("//div[@class='clearfix']//input")
    find_input.send_keys("Java")
    find_input.send_keys(Keys.ENTER)


if __name__ == '__main__':
    login("18844993804", "1130wang1130")
    find()
    # browser.quit()
