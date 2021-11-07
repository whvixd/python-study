import unittest
import matplotlib.pyplot as plt
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_plt(self):
        plt.plot(list(i for i in range(100)), list(j for j in range(100)), 'o')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def test_plt_1(self):
        x = np.arange(0, 4 * np.pi, 0.1)
        y = np.sin(x)
        z = np.cos(x)
        plt.plot(x, y, x, z)
        plt.show()

    def test_plt_2(self):
        x = np.array([6, 2, 13, 10])
        plt.plot(x, marker='o', ms=20, mec='r', linewidth=1.25)
        plt.show()

    def test_plt_scatter(self):
        # 1. figure 初试画布 2. add_subplot添加子图

        # 散点图
        plt.scatter(list(2010 + i for i in range(10)), list(j for j in range(10)))
        plt.show()
        pass

    def test_bar(self):
        # 条状图
        plt.bar([1, 2, 3, 4, 5], 0.2)
        plt.show()


if __name__ == '__main__':
    unittest.main()
