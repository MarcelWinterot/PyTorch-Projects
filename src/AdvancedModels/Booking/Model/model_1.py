import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.utils import BilinearComposition


class MLPBlock(nn.Module):
    def __init__(self, activation, dropout=0.0, use_norm=True):
        super(MLPBlock, self).__init__()
        self.activation = activation
        self.use_norm = use_norm

        self.linear_1 = nn.Linear(2304, 4096)
        self.linear_2 = nn.Linear(4096, 8192)
        self.linear_3 = nn.Linear(8192, 11987)

        if use_norm:
            self.norm_1 = nn.BatchNorm1d(9)
            self.norm_2 = nn.BatchNorm1d(9)
            self.norm_3 = nn.BatchNorm1d(9)

        self.drop = nn.Dropout(dropout)

    def forward(self, X):
        X = self.linear_1(X)
        X = self.activation(X)
        X = self.drop(X)

        if self.use_norm:
            X = self.norm_1(X)

        X = self.linear_2(X)
        X = self.activation(X)
        X = self.drop(X)

        if self.use_norm:
            X = self.norm_2(X)

        X = self.linear_3(X)

        return X


class RNNBlock(nn.Module):
    def __init__(self, activation, in_channels, in_between_channels, out_channels, num_layers=1, bidirectional=False, dropout=0.0, use_norm=True):
        super(RNNBlock, self).__init__()
        self.activation = activation
        self.use_norm = use_norm

        if bidirectional:
            self.lstm = nn.LSTM(
                in_channels, in_between_channels // 2, num_layers, batch_first=True, bidirectional=bidirectional)
            self.gru = nn.GRU(
                in_between_channels, out_channels // 2, num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(
                in_channels, in_between_channels, num_layers, batch_first=True, bidirectional=bidirectional)
            self.gru = nn.GRU(
                in_between_channels, out_channels, num_layers, batch_first=True, bidirectional=bidirectional)

        if use_norm:
            self.norm = nn.LayerNorm(out_channels)

        self.drop = nn.Dropout(dropout)

    def forward(self, X):
        X, _ = self.lstm(X)
        X = self.activation(X)
        X = self.drop(X)

        X, _ = self.gru(X)
        X = self.activation(X)
        X = self.drop(X)

        if self.use_norm:
            X = self.norm(X)

        return X


class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        # Simple RNN using Bidirectional LSTM GRU
        self.activation = nn.PReLU()
        self.rnn_norm = False

        self.rnn_1 = RNNBlock(self.activation, 1, 32, 64, num_layers=1, bidirectional=True,
                              dropout=0.0, use_norm=self.rnn_norm)
        self.rnn_2 = RNNBlock(self.activation, 64, 96, 128, num_layers=1, bidirectional=True,
                              dropout=0.0, use_norm=self.rnn_norm)
        self.rnn_3 = RNNBlock(self.activation, 128, 256, 256, num_layers=1, bidirectional=True,
                              dropout=0.0, use_norm=self.rnn_norm)
        self.rnn_4 = RNNBlock(self.activation, 256, 256, 256, num_layers=1, bidirectional=True,
                              dropout=0.0, use_norm=self.rnn_norm)
        self.rnn_5 = RNNBlock(self.activation, 256, 256, 256, num_layers=1, bidirectional=True,
                              dropout=0.0, use_norm=self.rnn_norm)

        self.rnns = nn.ModuleList(
            [self.rnn_1, self.rnn_2, self.rnn_3, self.rnn_4, self.rnn_5])

        self.flatten = nn.Flatten()

        self.bilinear = BilinearComposition(self.activation, MLPBlock, dropout=0.0,
                                            use_norm=False)

        self.softmax = nn.Softmax(dim=1)
        self.drop_02 = nn.Dropout(0.2)
        self.drop_05 = nn.Dropout(0.5)

        self.city_embedding = nn.Embedding(11987, 1)
        self.country_embedding = nn.Embedding(195, 1)

    def forward(self, X):
        cities = F.one_hot(X[:, 0].long(), num_classes=11987).squeeze(1)

        X[:, 0] = self.city_embedding(X[:, 0].long()).squeeze(2)
        X[:, 1] = self.country_embedding(X[:, 1].long()).squeeze(2)
        X[:, 2] = self.country_embedding(X[:, 2].long()).squeeze(2)

        for rnn in self.rnns:
            X = rnn(X)

        X = self.flatten(X)

        X = self.bilinear(X, cities)

        return X
