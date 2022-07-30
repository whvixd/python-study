import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import d2lzh_pytorch as d2l
import pandas as pd


# https://blog.csdn.net/Cyril_KI/article/details/125439045

# todo 转成torch数据格式
def loan_data():
    spectral_data = pd.read_csv(
        "/Users/whvixd/Documents/individual/MODIS/dataset/SL/spectral/h2o_data_withMissingS_0724.csv")

    need_features = spectral_data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]]
    # 将离散数值转变成指示特征
    need_features = pd.get_dummies(need_features, dummy_na=True)

    line_num = need_features.shape[0]
    train_line_num = int(line_num * 0.8)
    train_data = need_features.iloc[0:train_line_num]
    test_data = need_features.iloc[train_line_num:line_num]
    # 标准化
    # numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

    # 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
    # all_features[numeric_features] = all_features[numeric_features].fillna(0)


    # 分成训练集和测试集
    return train_data, test_data

class Arg:
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, output_size, dropout):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout


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
                            dropout=args.dropout,
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
    lr, dropout, num_epochs = 0.001, 0.75, 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = torch.nn.MSELoss()

    train_data, test_data = loan_data()
    net = CNN_LSTM(Arg(13, 32, 3, 4, 5, dropout))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    d2l.train_ch5(net, train_data, test_data, 0, optimizer, device, num_epochs, loss)
    # net.args={"in_channels":7,"out_channels":6,"hidden_size":3,"num_layers":5}
    # net.args.in_channels=7
    # print(net.args.num_layers)
