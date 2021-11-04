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


if __name__ == '__main__':
    unittest.main()
