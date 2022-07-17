import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import d2lzh_pytorch as d2l

# https://blog.csdn.net/Cyril_KI/article/details/125439045

class CNN_LSTM(nn.Module):
    def __int__(self, args):
        super(CNN_LSTM, self).__int__()
        self.args = args
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.in_channels,
                      out_channels=args.out_channels,
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
