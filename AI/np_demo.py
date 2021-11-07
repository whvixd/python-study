import unittest
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_np(self):
        arange_demo = np.arange(0, 1, 0.1, dtype=float)
        print(arange_demo)
        # 创建一位数组
        linspace_demo = np.linspace(1, 100, num=10)
        print(linspace_demo)

        # 等比数列 10**0,10**2
        logspace_demo = np.logspace(0, 2, 20)

        # 索引
        print(logspace_demo[1])

        # 生成2X3的零矩阵
        np.zeros((2, 3), dtype=float)

        # 生成3X5的矩阵，k=0,为对角数为1
        print(np.eye(3, 5, k=0))

        # 生成 3X3矩阵，对角线为 1，3，5
        print("1d np.diag :\n", np.diag([1, 3, 5]))
        # 2维数组，返回对角线的值
        print("2d np.diag:\n", np.diag([[1, 2, 3],
                                        [1, 5, 3],
                                        [1, 2, 7]]))

        # 输入一个100个随机数组
        print(np.random.random(100))

        # 随机生成10X5的随机正态分布矩阵,ravel展平
        randn_demo = np.random.randn(10, 5)
        print(randn_demo.ravel())
        print(randn_demo.flatten(order='F'))

        #         矩阵的创建
        np.mat("123;456;789")

        # pandas处理表格，json相关数据结构

    def test_slice(self):
        np_array = np.array([[i for i in range(5)], [j + 10 for j in range(5)]])
        print(np_array[1, :3])

        # 改变数组形状
        b = np.arange(24).reshape(2, 3, 4)
        print(b)

        # 拆分数组，仅改变小形状
        print(b.ravel())

        # 拉直，新的数组
        print(b.flatten())

        # 转置
        print(b.transpose())

    def test_stack(self):
        a = np.arange(9).reshape(3, 3)
        b = a * 2
        # 垂直堆积
        print(np.vstack((a, b)))
        # 水平堆积
        print(np.hstack((a, b)))
        # 深度堆积
        print(np.dstack((a, b)))

    def test_split(self):
        a = np.arange(9).reshape(3, 3)
        # 横向拆分
        print(np.hsplit(a, 3))

        # 纵向拆分
        print(np.vsplit(a, 3))

        # 深度拆分，要求矩阵的秩>=3
        c = np.arange(27).reshape(3, 3, 3)
        print(np.dsplit(c, 3))


if __name__ == '__main__':
    unittest.main()
