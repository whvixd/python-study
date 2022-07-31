import unittest

import torch
from torch import nn


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_conv1d(self):
        net = nn.Conv1d(16, 33, 3, stride=2)
        input = torch.randn(20, 16, 50)
        print("input:", input)
        output = net(input)
        print("output:", output)
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
