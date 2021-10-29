from typing import Optional, List

'''
https://docs.python.org/zh-cn/3/library/typing.html
Python 运行时不强制执行函数和变量类型注解，但这些注解可用于类型检查器、IDE、静态检查器等第三方工具。
'''


def list_(b: Optional[List[int]] = None):
    print(b)


list_([1, 2, 3])
