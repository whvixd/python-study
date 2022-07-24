import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import d2lzh_pytorch as d2l
import pandas as pd


# https://blog.csdn.net/Cyril_KI/article/details/125439045

def data_process():
    spectral_data = pd.read_csv(
        "/Users/whvixd/Documents/individual/MODIS/dataset/SL/spectral/h2o_data_withMissingS_0724.csv")

    all_features = spectral_data.iloc[0:10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]]
    print(all_features[0:10])
    # 标准化
    # numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

    # 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
    # all_features[numeric_features] = all_features[numeric_features].fillna(0)
    print(all_features[0:10])
    # todo 分成训练集和测试集
    return all_features, all_features


def train():
    lr = 0.01
    net = CNN_LSTM(Arg(13, 7, 3, 4, 5))
    # todo 参数待调试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_data, test_data = data_process()

    # todo 数据保存，多线程处理
    d2l.train_ch5(net, train_data, test_data, 100, optimizer, device, 100, loss=torch.nn.MSELoss())


class Arg:
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, output_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size


class CNN_LSTM(nn.Module):
    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.args = args
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.in_channels,
                      out_channels=args.out_channels,
                      # 3个卷积核，三个纬度的特征
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )

        self.lstm = nn.LSTM(input_size=args.out_channels,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            batch_first=True)

        self.fc = nn.Linear(args.hidden_size, args.output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)

        x = x[:, -1, :]
        return x


if __name__ == '__main__':
    # data_process()
    net = CNN_LSTM(Arg(1, 2, 3, 4, 5))
    # net.args={"in_channels":7,"out_channels":6,"hidden_size":3,"num_layers":5}
    # net.args.in_channels=7
    print(net.args.num_layers)
