import unittest
from d2lzh_pytorch import *
from cnn_demo import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_Conv2D(self):
        # 0为黑色，1为白色
        x = torch.ones(6, 8)
        x[:, 2:6] = 0

        # 白到黑的边缘用1表示，黑到白的边缘用-1表示
        k = torch.Tensor([[1, -1]])
        y = d2l.corr2d(x, k)

        # 现在利用输入x和输出y来学习构造我们的核数组k

        # 构造一个核数组形状是(1, 2)的二维卷积层

        conv2d = Conv2D((1, 2))
        step = 20
        lr = 0.01
        for i in range(step):
            y_hat = conv2d(x)

            # 平方差损失函数
            l = ((y_hat - y) ** 2).sum()

            l.backward()

            # 梯度下降
            conv2d.weight.data -= lr * conv2d.weight.grad
            conv2d.bias.data -= lr * conv2d.bias.grad

            # 梯度清零
            conv2d.weight.grad.zero_()
            conv2d.bias.grad.fill_(0)

            print("i:%d,loss:%.3f" % (i + 1, l.item()))

        print("weight: ", conv2d.weight.data)
        print("bias: ", conv2d.bias.data)

    def test_LeNet(self):
        from net_demo import LeNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = LeNet()
        print(net)

        batch_size = 256
        train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    def test_AlexNet(self):
        from net_demo import AlexNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = AlexNet()
        print(net)
        batch_size = 128
        # 如出现“out of memory”的报错信息，可减小batch_size或resize
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
        # 训练
        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    def test_VGG(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        # 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
        fc_features = 512 * 7 * 7  # c * w * h
        fc_hidden_units = 4096  # 任意

        net = vgg(conv_arch, fc_features, fc_hidden_units)
        X = torch.rand(1, 1, 224, 224)

        # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
        for name, blk in net.named_children():
            X = blk(X)
            print(name, 'output shape: ', X.shape)

        # 因为VGG-11计算上比AlexNet更加复杂，出于测试的目的我们构造一个通道数更小，或者说更窄的网络在Fashion-MNIST数据集上进行训练。
        ratio = 8
        small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio),
                           (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio)]
        net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
        print(net)

        batch_size = 64
        # 如出现“out of memory”的报错信息，可减小batch_size或resize
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    def test_NiN(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = nn.Sequential(
            nin_block(1, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            # 标签类别数10
            nin_block(384, 10, kernel_size=3, stride=1, padding=1),
            GlobalAvgPool2d(),
            # 将四维的输出转乘二维的输出，其形状为（批量大小10）
            FlattenLayer()
        )

        X = torch.rand(1, 1, 224, 224)
        for name, blk in net.named_children():
            X = blk(X)
            print(name, 'output shape: ', X.shape)
        batch_size = 128
        # 如出现“out of memory”的报错信息，可减小batch_size或resize
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

        lr, num_epochs = 0.002, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    def test_GoogLeNet(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from net_demo import Inception
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                           nn.Conv2d(64, 192, kernel_size=3, padding=1),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                           Inception(256, 128, (128, 192), (32, 96), 64),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                           Inception(512, 160, (112, 224), (24, 64), 64),
                           Inception(512, 128, (128, 256), (24, 64), 64),
                           Inception(512, 112, (144, 288), (32, 64), 64),
                           Inception(528, 256, (160, 320), (32, 128), 128),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                           Inception(832, 384, (192, 384), (48, 128), 128),
                           GlobalAvgPool2d())

        net = nn.Sequential(b1, b2, b3, b4, b5, FlattenLayer(), nn.Linear(1024, 10))
        X = torch.rand(1, 1, 96, 96)
        for blk in net.children():
            X = blk(X)
            print('output shape: ', X.shape)

        batch_size = 128
        # 如出现“out of memory”的报错信息，可减小batch_size或resize
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)

        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    # https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.10_batch-norm
    def test_BatchNorm(self):
        from net_demo import BatchNorm
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            BatchNorm(6, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            BatchNorm(16, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            FlattenLayer(),
            nn.Linear(16 * 4 * 4, 120),
            BatchNorm(120, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(84, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        batch_size = 256
        train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
        # 最后我们查看第一个批量归一化层学习到的拉伸参数gamma和偏移参数beta
        net[1].gamma.view((-1,)), net[1].beta.view((-1,))

    def test_torch_BatchNorm(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),  # 用于卷积层 num_features
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            FlattenLayer(),
            nn.Linear(16 * 4 * 4, 120),
            nn.BatchNorm1d(120),  # # 用于全连接层 num_features
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        batch_size = 256
        train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    def test_Residual(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
        net.add_module("resnet_block2", resnet_block(64, 128, 2))
        net.add_module("resnet_block3", resnet_block(128, 256, 2))
        net.add_module("resnet_block4", resnet_block(256, 512, 2))

        net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
        net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))

        X = torch.rand((1, 1, 224, 224))
        for name, layer in net.named_children():
            X = layer(X)
            print(name, ' output shape:\t', X.shape)

        batch_size = 256
        # 如出现“out of memory”的报错信息，可减小batch_size或resize
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)

        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    # 稠密层：https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.12_densenet
    def test_DenseNet(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        blk = DenseBlock(2, 3, 10)
        X = torch.rand(4, 3, 8, 8)
        Y = blk(X)
        print(Y.shape)  # torch.Size([4, 23, 8, 8])

        blk = transition_block(23, 10)
        print(blk(Y).shape)  # torch.Size([4, 10, 4, 4])

        net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
        num_convs_in_dense_blocks = [4, 4, 4, 4]

        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            DB = DenseBlock(num_convs, num_channels, growth_rate)
            net.add_module("DenseBlock_%d" % i, DB)
            # 上一个稠密块的输出通道数
            num_channels = DB.out_channels
            # 在稠密块之间加入通道数减半的过渡层
            if i != len(num_convs_in_dense_blocks) - 1:
                net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        net.add_module("BN", nn.BatchNorm2d(num_channels))
        net.add_module("relu", nn.ReLU())
        net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
        net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10)))

        X = torch.rand((1, 1, 96, 96))
        for name, layer in net.named_children():
            X = layer(X)
            print(name, ' output shape:\t', X.shape)

        batch_size = 256
        # 如出现“out of memory”的报错信息，可减小batch_size或resize
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

        lr, num_epochs = 0.001, 5
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    unittest.main()
