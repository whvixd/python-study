import unittest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_boxplot(self):
        # 读取数据集
        train_data = pd.read_csv('../../test/sources/data/kaggle_house/train.csv')
        test_data = pd.read_csv('../../test/sources/data/kaggle_house/test.csv')

        year_var = 'YearBuilt'
        sale_var='SalePrice'
        data = pd.concat([train_data[sale_var], train_data[year_var]], axis=1)
        # f, ax = plt.subplots(figsize=(26, 12))
        fig = sns.boxplot(x=year_var, y=sale_var, data=data)
        fig.axis(ymin=0, ymax=800000)



if __name__ == '__main__':
    unittest.main()
