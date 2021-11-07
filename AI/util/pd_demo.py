import unittest
from pandas import Series, DataFrame
import pandas as pd
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_Data_Frame(self):
        dictionary = {'state': ['0hio', '0hio', '0hio', 'Nevada', 'Nevada'],
                      'year': [2000, 2001, 2002, 2001, 2002],
                      'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
        frame = DataFrame(dictionary)
        # 修改列名
        frame = DataFrame(dictionary, index=['one', 'two', 'three', 'four', 'five'])

        frame['add'] = [0, 0, 0, 0, 0]



    def test_sort(self):
        # Series用sort_index()
        # 按索引排序，sort_values()
        # 按值排序；
        # DataFrame也是用sort_index()
        # 和sort_values()。
        obj = Series(range(4), index=['d', 'a', 'b', 'c'])

        print(obj)
        print(obj.sort_index())

        frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
        # 按行排序
        frame.sort_index(axis=1, ascending=False)

        # 按值排序
        print(obj.sort_values())

        frame.sort_values(by='b')  # DataFrame必须传一个by参数表示要排序的列

    def test_delete(self):
        s1=Series([i for i in range(5)],index=['a','b','c','d','e'])
        s1.drop('c')

        df=DataFrame(np.arange(9).reshape(3, 3), index=['a', 'c', 'd'], columns=['oh', 'te', 'ca'])
        # 删除a行
        df.drop('a')
        # 删除列
        df.drop(['oh', 'te'], axis=1)

        # 需要注意的是drop()返回的是一个新对象，原对象不会被改变。

    def test_operate(self):
        d1=DataFrame(np.arange(12).reshape((3,4)),columns=list('abcd'))
        d2=DataFrame(np.arange(20).reshape((4,5)),columns=list('abcde'))
        print(d1+d2)

        print(d1.add(d2,fill_value=0))

    def test_duplicated(self):
        df = DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [1, 1, 2, 3, 3, 4, 4]})
        # DataFrame的duplicated方法返回一个布尔型Series，表示各行是否是重复行
        print(df.duplicated())

        # 用于去除重复的行数
        print(df.drop_duplicates())

    def test_hierarchical(self):
        # 层次化索引(hierarchical indexing)是pandas的一项重要功能，它使我们能在一个轴上拥有多个（两个以上）索引级别
        data = Series(np.random.randn(10),
                      index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'], [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])

        print(data['b':'d'])

        print(data[:, 2])

        # 将Series转化成DataFrame
        df=data.unstack()


if __name__ == '__main__':
    unittest.main()
