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

    def test_save(self):
        years = [year for year in range(2000, 2016)]

        values = [12914, 11826, 12997, 12306.41, 12327.28, 11406, 10608, 8378, 8667.02, 8052.78, 6922.52, 5744, 4196,
                  4336, 4588, 4751]

        plt.bar(years, values[::-1], color='#800080')  # 绘制柱状图，alpha 表示颜色的透明程度
        plt.yticks([key for key in range(4000, 14000, 1000)])
        plt.xticks(years, rotation=45)
        plt.axis(ymin=4000, ymax=14000)
        plt.show()  # 展示图像

    def test_bar1(self):
        xstring = '2015 2014 2013 2012 2011     \
                   2010 2009 2008 2007 2006     \
                   2005 2004 2003 2002 2001    2000'  # x轴标签
        n = 6
        ystring = [''] * n  # y轴对应的6组数据
        ystring[
            0] = '6793    6324    6237    5790.99    5357.1    5032    4681    3800    3863.9    3366.79    3167.66    2778    2359    2250    2170    2112'
        ystring[
            1] = '6473    5933    5850    5429.93    4993.17    4725    4459    3576    3645.18    3119.25    2936.96    2608    2197    2092    2017    1948'
        ystring[
            2] = '15157    12965    12591    11460.19    10993.92    10934    9662    7801    7471.25    6584.93    5833.95    5576    4145    4154    4348    4288'
        ystring[
            3] = '12914    11826    12997    12306.41    12327.28    11406    10608    8378    8667.02    8052.78    6922.52    5744    4196    4336    4588    4751'
        ystring[
            4] = '9566    9817    9777    9020.91    8488.21    7747    6871    5886    5773.83    5246.62    5021.75    3884    3675.14    3488.57    3273.53    3260.38'
        ystring[
            5] = '4845    5177    4907    4305.73    4182.11    4099    3671    3219    3351.44    3131.31    2829.35    2235    2240.74    1918.83    2033.08    1864.37'
        labels = ['Commercial housing', 'Residential commercial housing',
                  'high-end apartments', 'Office Building', 'Business housing', 'Others']  # 图例标签
        colors = ['#ff7f50', '#87cefa', '#DA70D6', '#32CD32', '#6495ED', '#FF69B4']  # 指定颜色
        #  请在此添加实现代码  #
        # ********** Begin *********#
        xlabels = xstring.split()  # 年份切分
        xlabels.reverse()  # 年份序列倒序排列，从小到大
        x = np.arange(1, n * len(xlabels), n)  # x轴条形起始位置
        w = 0.8  # 条形宽度设置
        for i in range(n):
            y = ystring[i].split()
            y.reverse()
            y = [float(e) for e in y]  # 将划分好的字符串转为float类型
            plt.bar(x + i * w, y, width=w, color=colors[i])  # 以指定颜色绘制柱状图
        plt.ylim([1450, 15300])  # 指定y轴范围
        plt.yticks(range(2000, 15000, 2000))  # 指定y轴刻度
        plt.xlim([-1, 98])
        plt.xticks(x + w * 2.5, xlabels, rotation=45)  # 添加x轴标签，旋转40度
        plt.legend(labels, loc='upper left')  # 添加图例，位置为左上角
        plt.title('Selling Prices of Six Types of Housing')
        plt.show()  # 展示图像
        # ********** End **********#


if __name__ == '__main__':
    unittest.main()
