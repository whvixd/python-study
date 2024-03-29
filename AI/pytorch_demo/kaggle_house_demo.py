import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import d2lzh_pytorch as d2l

torch.set_default_tensor_type(torch.FloatTensor)

# 读取数据集
train_data = pd.read_csv('../../test/sources/data/kaggle_house/train.csv')
test_data = pd.read_csv('../../test/sources/data/kaggle_house/test.csv')

# 共1460个样本和80个特征
print(train_data.shape)  # 输出 (1460, 81)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])  # 后面的数组是指定输出的列
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # 第一列是id，不需要，训练集最后一个是标签

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 将离散数值转变成指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)  # (2919,331) --> 将字段的枚举类型单独作为一特征，所列多了

# 转Numpy格式
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

# 训练模型（线性回归+平方差损失函数）
loss = torch.nn.MSELoss()


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for p in net.parameters():
        nn.init.normal_(p, mean=0, std=0.01)
    return net


# 定义对数均方根误差：
def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 返回第i折交叉验证时所需要的训练和验证的数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k  # k子集，分k个子集数据 。// 地板除法，向下取整
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 子集
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


# 在K折交叉验证中我们训练K次并返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs, lr, wd, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, lr, wd, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls, ['epochs', 'valid'])
        print("fold %d,train rmse %f, valid rmse %f" % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64


train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

# 训练并预测房价
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, wd, batch_size):
    # net为线性回归
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, wd, batch_size)
    # print("parameters:", list(net.parameters()))
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    pred = net(test_features).detach().numpy()  # detach() 切断向前传播，requires_grad=false
    test_data['SalePrice'] = pd.Series(pred.reshape(1, -1)[0])
    submission = pd.concat([test_data["Id"], test_data['SalePrice']], axis=1)
    submission.to_csv('../../test/sources/data/kaggle_house/submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
