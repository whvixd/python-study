import numpy as np


def split_seq(train_data, seq_size):
    X, y = [], []
    for i in range(len(train_data)):
        cur_p = i + seq_size
        if cur_p > len(train_data) - 1:
            break
        train_x = train_data[i:cur_p]
        train_y = train_data[cur_p]
        X.append(train_x)
        y.append(train_y)

    np_X, np_y = np.array(X), np.array(y)
    return np_X, np_y
