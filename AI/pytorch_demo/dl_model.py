import torch
import torch.nn as nn


# https://blog.csdn.net/weixin_41744192/article/details/115270178
# https://zhuanlan.zhihu.com/p/102904450

class LSTM_FC(nn.Module):
    def __init__(self):
        super(LSTM_FC, self).__init__()
        '''
        参数有：
    input_size：x的特征维度
    hidden_size：隐藏层的特征维度
    num_layers：lstm隐层的层数，默认为1
    bias：False则bihbih=0和bhhbhh=0. 默认为True
    batch_first：True则输入输出的数据格式为 (batch, seq, feature)
    dropout：除最后一层，每一层的输出都进行dropout，默认为: 0
    bidirectional：True则为双向lstm默认为False
如下：
第一维度是样本数（城市，地点），第二维度是时间（日期），第三维度是特征（多变量）

    t1:
 城市,v1,v2,v3
 上海,1,2,1
 北京,3,2,1
 安徽,4,5,6

  t2:
 城市,v1,v2,v3
 上海,2,3,2
 北京,2,1,2
 安徽,2,1,6

 batch_first:True

 城市: 
 t1,t2,t3,t4
 上海,上海,...
 北京,北京
 安徽,安徽

  v1：
 t1,t2,t3,t4
 1,2
 3,2
 4,2

    '''
        self.rnn = nn.LSTM(input_size=3,  # 特征长度
                           hidden_size=5,  # 隐藏层特征维度
                           num_layers=2  # 隐藏层层数
                           )

        self.reg = nn.Sequential(
            nn.Linear(5, 1)
        )

    def forward(self, x):
        '''
        input(seq_len, batch, input_size)
    参数有：
    seq_len：序列长度，在NLP中就是句子长度，一般都会用pad_sequence补齐长度
    batch：每次喂给网络的数据条数，在NLP中就是一次喂给网络多少个句子
    input_size：特征维度，和前面定义网络结构的input_size一致。
    六个样本：1,2,3,4,5,6
    seq_len:3
    1-2-3,2-3-4,3-4-5...
    batch:2
    第一次:1-2-3,2-3-4
    第一次:2-3-4,3-4-5
        '''

        y, (ht, ct) = self.rnn(x)

        '''
        (2,5,5)
        ht(num_layers * num_directions, batch, hidden_size) 最后一个状态的隐含层的状态值
        ct(num_layers * num_directions, batch, hidden_size) 最后一个状态的隐含层的遗忘门值

        num_layers：隐藏层数
        num_directions：如果是单向循环网络，则num_directions=1，双向则num_directions=2
        batch：输入数据的batch 样本数
        hidden_size：隐藏层神经元个数
        '''
        print("ht.size:", ht.shape)  # 2,5,5
        print("ct.size:", ct.shape)  # 2,5,5
        seq_len, batch_size, hidden_size = y.shape
        x = y.view(-1, hidden_size)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x


class CNN_conv1d(nn.Module):
    def __init__(self):
        super(CNN_conv1d, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 如果是20个时刻的7特征数据

        self.conv1 = nn.Sequential(
            # 卷积后20-2+1=19
            nn.Conv1d(in_channels=7, out_channels=15, kernel_size=2),
            nn.ReLU(),
            # 卷积后19-2+1=18
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.conv2 = nn.Sequential(
            # 卷积后18-2+1=17
            nn.Conv1d(in_channels=15, out_channels=31, kernel_size=2),
            nn.ReLU(),
            # 卷积后17-2+1=16
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.linear1 = nn.Linear(in_features=31 * 16, out_features=25)
        self.linear2 = nn.Linear(in_features=25, out_features=1)  # 预测下一时刻的数据

    def forward(self, x):
        x = self.conv1(x)  # [batch_size, in_channels, in_size]
        x = self.conv2(x)
        x = x.view(-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(x.shape[0], -1)
        return x


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class CNN_conv2d(nn.Module):
    def __init__(self):
        super(CNN_conv2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=15, kernel_size=3),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=31, kernel_size=3),
            nn.BatchNorm2d(31),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        # https://www.cnblogs.com/douzujun/p/13366939.html
        self.linear = nn.Sequential(
            FlattenLayer(),
            nn.Linear(in_features=31*11*11, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=32, out_features=1) # 输出后的y和真实值对比（对于需要hXw 如何处理）
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        print(x.shape)
        y = self.linear(x)
        return y
