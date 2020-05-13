#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

a = 1
b = 2

c = 3.33
d = 4.0

# complex() 鍑芥暟鐢ㄤ簬鍒涘缓涓�涓�间负 real + imag * j 鐨勫鏁版垨鑰呰浆鍖栦竴涓瓧绗︿覆鎴栨暟涓哄鏁般�傚鏋滅涓�涓弬鏁颁负瀛楃涓诧紝鍒欎笉闇�瑕佹寚瀹氱浜屼釜鍙傛暟銆�
e = complex(float(a), float(b))

print("a's type is :", type(a))
print("e'type is :", type(e))
print(complex(a, b))
print(sys.float_info)
print("王志祥\n你好")
