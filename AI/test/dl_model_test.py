import unittest

from .. pytorch_demo .dl_model import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_lstm(self):
        # 在序列模型时，需要将原始数据进行处理
        # 原始数据(100,3)
        # 转成LSTM的输入模型(100*10*10,3)，第一循环取前(0:batch_size)数据，第二个循环取(batch_size:2*batch_size),所以batch=100*10*10/batch_size+1
        input = torch.randn(10, 6, 3)  # (seq_len, batch, input_size) 10:时间长度，6:样本个数,3:样本特征维度
        print("input.shape:", input.shape)
        model = LSTM_FC()
        output = model.forward(input)
        print("output.shape:", output.shape)
        print("output:", output)

    def test_cnn_1d(self):
        input = torch.randn(20, 7)
        print(input.shape)
        # 扩充一维，作为batch_size
        input = input.unsqueeze(0)
        print(input.shape)
        # 调换位置
        # nn.Conv1d对输入数据的最后一维进行一维卷积，需要调换第二维和第三维数据
        input = input.permute(0, 2, 1)
        print(input.shape)

        model = CNN_conv1d()
        output = model.forward(input)
        print(output)

    def test_cnn_2d(self):
        input = torch.randn(1, 7, 111, 110) # (batch_size,channel,h,w)
        print(input.shape)

        model = CNN_conv2d()
        output = model.forward(input)
        print(output)

    def test_cnn_time(self):
        train_seq=[i for i in range(20)]
        seq_time=3
        feature=1

        for i in range(len(train_seq)-seq_time):
            train_x=train_seq[i:i+seq_time]
            train_x=train_seq[i+seq_time:i+seq_time+1]




if __name__ == '__main__':
    unittest.main()
