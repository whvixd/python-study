#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import requests

print(os.getcwd())

r = requests.get("http://www.qilixiang1118.top")
print(r.url)
print(r.encoding)
print(r.text)
