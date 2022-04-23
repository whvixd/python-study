import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here


    def test_01(self):
        l_=[1,1,3,4]

        t_=l_[0<0.1]

        print(t_)

        self.assertEqual(True, True)  # add assertion here

if __name__ == '__main__':
    unittest.main()
