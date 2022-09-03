import unittest
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.optim as optim
from ..pytorch_demo.dl_model import *
from ..pytorch_demo.d2lzh_pytorch import *
from ..util import data_process_util
import numpy as np


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
        input = torch.randn(1, 7, 111, 110)  # (batch_size,channel,h,w)
        print(input.shape)

        model = CNN_conv2d()
        output = model.forward(input)
        print(output)

    def test_cnn_time(self):
        train_seq = [i for i in range(20)]
        seq_time = 3
        feature = 1

        train_seq = np.array(train_seq, dtype=np.float32).reshape(len(train_seq), 1)

        # 加了归一化模型效果还较差
        # 标准化处理，加速收敛
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler(feature_range=(0, 2))
        # train_seq = scaler.fit_transform(train_seq)  # 输入数据结构二维 (n,1)
        # print(train_seq.shape)

        np_X, np_y = data_process_util.split_seq(train_seq, seq_time)
        model = CNN_conv1d_time_seq()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        train_dataset = TorchDataset(np_X, np_y)
        train_loader = DataLoader(dataset=train_dataset)

        for i in range(100):
            total_loss = 0
            for index, (input, label) in enumerate(train_loader):
                # input = input.unsqueeze(0)
                input = input.permute(0, 2, 1)
                pred = model(Variable(input))

                label=label.squeeze(1)
                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print("epoch:%d,loss=%f" % (i, total_loss))


#         预测
        model.eval()
        test_X=[30,31,32]
        np_X=np.array(test_X,dtype=np.float32).reshape(1,3)
        input_X=torch.from_numpy(np_X)

        input_X=input_X.unsqueeze(0)

        pre=model(Variable(input_X))
        pre_rest=pre.data.squeeze(1)
        print(pre_rest)


if __name__ == '__main__':
    unittest.main()
