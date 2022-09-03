import unittest
from d2lzh_pytorch import *
from rnn_demo import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_one_hot(self):
        print(one_hot(torch.tensor([1, 3]), 10))

    def test_rnn(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
        num_hiddens = 256
        X = torch.arange(10).view(2, 5)
        inputs = to_onehot(X, vocab_size)
        print(len(inputs), inputs[0].shape)

        state = init_rnn_state(X.shape[0], num_hiddens, device)
        inputs = to_onehot(X.to(device), vocab_size)
        params = get_params(vocab_size)
        outputs, state_new = rnn(inputs, state, params)
        print(len(outputs), outputs[0].shape, state_new[0].shape)

        predict_res = predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
                                  device, idx_to_char, char_to_idx)
        print(predict_res)

        num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
        pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

        train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                              vocab_size, device, corpus_indices, idx_to_char,
                              char_to_idx, True, num_epochs, num_steps, lr,
                              clipping_theta, batch_size, pred_period, pred_len,
                              prefixes)

    # https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter06_RNN/6.5_rnn-pytorch
    def test_pytorch_rnn(self):
        (corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
        num_hiddens = 256
        # rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens) # 已测试
        rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

        num_steps = 35
        batch_size = 2
        state = None
        X = torch.rand(num_steps, batch_size, vocab_size)
        Y, state_new = rnn_layer(X, state)
        print(Y.shape, len(state_new), state_new[0].shape)

        model = RNNModel(rnn_layer, vocab_size).to(device)
        print(predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))

        num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2  # 注意这里的学习率设置
        pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
        train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                      corpus_indices, idx_to_char, char_to_idx,
                                      num_epochs, num_steps, lr, clipping_theta,
                                      batch_size, pred_period, pred_len, prefixes)

    def test_GRU(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        (corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
        num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

        num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
        pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

        train_and_predict_rnn(gru,get_gru_params,init_gru_state,num_hiddens,vocab_size,device,corpus_indices,idx_to_char,
                              char_to_idx,False,num_epochs,num_steps,lr,clipping_theta,batch_size,pred_period,pred_len,prefixes)

    def test_simple_GRU(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        (corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
        num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
        gru=nn.GRU(input_size=vocab_size,hidden_size=num_hiddens)
        model=RNNModel(gru,vocab_size).to(device)
        num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
        pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

        train_and_predict_rnn(model,get_gru_params,init_gru_state,num_hiddens,vocab_size,device,corpus_indices,idx_to_char,
                              char_to_idx,False,num_epochs,num_steps,lr,clipping_theta,batch_size,pred_period,pred_len,prefixes)

    def test_data_iter_consecutive(self):
        my_seq = list(range(100))
        for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
            print('X: ', X, '\nY:', Y, '\n')


if __name__ == '__main__':
    unittest.main()
