import unittest

from ..util import data_process_util


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_split_seq(self):
        data=[i for i in range(20)]

        np_X, np_y=data_process_util.split_seq(data,5)
        print(np_X)
        print(np_y)


if __name__ == '__main__':
    unittest.main()
