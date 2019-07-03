# coding=utf-8
class SiameseLSTMConfig(object):
    def __init__(self):
        self.maxlen = 15
        self.max_features = 7901
        self.embedding_size = 512
        self.lstm_dim = 256
        self.dense_dim = 256
        self.dropout_rate = 0.3
        self.batch_size = 1024
        self.epochs = 50
