#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by wangzhx on 2020/05/15

# 破解上学吧获取答案的次数限制
# 上学吧网址：https://www.shangxueba.com/ask
# 代码参考 wjszfq@CSDN 发布的源码，链接：https://blog.csdn.net/qq_41861526/article/details/85573479
# by Tsing 2019.03.26

import os
import random
import requests
import urllib3

urllib3.disable_warnings()  # 这句和上面一句是为了忽略 https 安全验证警告，参考：https://www.cnblogs.com/ljfight/p/9577783.html
from bs4 import BeautifulSoup
from PIL import Image


def get_verifynum(session):  # 网址的验证码逻辑是先去这个网址获取验证码图片，提交计算结果到另外一个网址进行验证。
    r = session.get("https://www.shangxueba.com/ask/VerifyCode2.aspx",
                    verify=False)  # HTTPS 请求进行 SSL 验证或忽略 SSL 验证才能请求成功，忽略方式为 verify=False。参考：https://www.cnblogs.com/ljfight/p/9577783.html
    with open('temp.png', 'wb+') as f:
        f.write(r.content)
    image = Image.open('temp.png')
    image.show()  # 调用系统的图片查看软件打开验证码图片，如果不能打开，可以自己找到 temp.png 打开。
    verifynum = input("\n请输入验证码图片中的计算结果：")
    image.close()
    os.remove("temp.png")
    return verifynum


def get_question(session):
    r = session.get(link)
    soup = BeautifulSoup(r.content, "html.parser")
    description = soup.find(attrs={"name": "description"})['content']  # 抓取题干内容
    return description


def get_answer(session, verifynum, dataid):
    data1 = {
        "Verify": verifynum,
        "action": "CheckVerify",
    }
    session.post("https://www.shangxueba.com/ask/ajax/GetZuiJia.aspx", data=data1)  # 核查验证码正确性
    data2 = {
        "phone": "",
        "dataid": dataid,
        "action": "submitVerify",
        "siteid": "1001",
        "Verify": verifynum,
    }
    r = session.post("https://www.shangxueba.com/ask/ajax/GetZuiJia.aspx", data=data2)
    soup = BeautifulSoup(r.content, "html.parser")
    ans = soup.find('h6')
    print("\n" + '-' * 45)
    if (ans):  # 只有验证码核查通过才会显示答案
        print("\n题目：" + get_question(session))
        print(ans.text)
    else:
        print('\n没有找到答案！请检查验证码或网址是否输入有误！\n')
    print('-' * 45)


if __name__ == '__main__':
    s = requests.session()
    while True:
        s.headers.update({"X-Forwarded-For": "%d.%d.%d.%d" % (
            random.randint(120, 125), random.randint(1, 200), random.randint(1, 200),
            random.randint(1, 200))})  # 这一句是整个程序的关键，通过修改 X-Forwarded-For 信息来欺骗 ASP 站点对于 IP 的验证。
        link = raw_input("\n请输入上学吧网站上某道题目的网址，例如：https://www.shangxueba.com/ask/8952241.html\n\n请输入：").strip()  # 过滤首尾的空格
        print "输入的网址:" + link
        if (link[0:31] != "https://www.shangxueba.com/ask/" or link[-4:] != "html"):
            print("\n网址输入有误！请重新输入！\n")
            continue
        dataid = link.split("/")[-1].replace(r".html", "")  # 提取网址最后的数字部分
        if (dataid.isdigit()):  # 根据格式，dataid 应该全部为数字，判断字符串是否全部为数字，返回 True 或者 False
            # todo 验证码有问题
            verifynum = get_verifynum(s)
            get_answer(s, verifynum, dataid)
        else:
            print("\n网址输入有误！请重新输入！\n")
            continue
