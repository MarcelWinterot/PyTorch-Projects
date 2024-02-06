import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.utils import MLPBlock


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
        self.rnn_drop = 0.0
        self.max_time_features = 256
        self.rnns_with_max_time_features = 2
        self.bidirectional = True

        self.mlp_norm = False
        self.mlp_drop = 0.0

        self.rnn_1 = RNNBlock(self.activation, 1, 32, 64, num_layers=1, bidirectional=self.bidirectional,
                              dropout=self.rnn_drop, use_norm=self.rnn_norm)
        self.rnn_2 = RNNBlock(self.activation, 64, 96, 128, num_layers=1, bidirectional=self.bidirectional,
                              dropout=self.rnn_drop, use_norm=self.rnn_norm)
        self.rnn_3 = RNNBlock(self.activation, 128, 256, self.max_time_features, num_layers=1, bidirectional=self.bidirectional,
                              dropout=self.rnn_drop, use_norm=self.rnn_norm)
        for i in range(self.rnns_with_max_time_features):
            setattr(self, f"rnn_{i+4}", RNNBlock(
                self.activation, self.max_time_features, self.max_time_features, self.max_time_features, num_layers=1, bidirectional=self.bidirectional, dropout=self.rnn_drop, use_norm=self.rnn_norm))

        self.rnns = nn.ModuleList(
            [self.rnn_1, self.rnn_2, self.rnn_3, *[
                getattr(self, f"rnn_{i+4}") for i in range(self.rnns_with_max_time_features)]
             ])

        self.flatten = nn.Flatten()

        self.mlp = MLPBlock(self.activation, dropout=self.mlp_norm,
                            use_norm=self.mlp_norm, last_layer=False, num_channels=self.max_time_features, num_labels=20)

        self.softmax = nn.Softmax(dim=1)
        self.drop_02 = nn.Dropout(0.2)
        self.drop_05 = nn.Dropout(0.5)

        self.city_embedding = nn.Embedding(11989, 1)
        self.country_embedding = nn.Embedding(196, 1)
        self.affiliate_embedding = nn.Embedding(10698, 1)
        self.device_embedding = nn.Embedding(3, 1)
        self.city_number_embedding = nn.Embedding(49, 1)
        self.year_embedding = nn.Embedding(3, 1)
        self.month_embedding = nn.Embedding(12, 1)
        self.day_embedding = nn.Embedding(31, 1)

        self.output_bias = nn.Parameter(torch.zeros(11989))

    def forward(self, X):
        X[:, 0] = self.city_embedding(X[:, 0].long()).squeeze(2)
        X[:, 1] = self.country_embedding(X[:, 1].long()).squeeze(2)
        X[:, 2] = self.country_embedding(X[:, 2].long()).squeeze(2)
        X[:, 3] = self.affiliate_embedding(X[:, 3].long()).squeeze(2)
        X[:, 4] = self.device_embedding(X[:, 4].long()).squeeze(2)
        X[:, 5] = self.city_number_embedding(X[:, 5].long()).squeeze(2)

        X[:, 6] = self.city_embedding(X[:, 6].long()).squeeze(2)
        X[:, 7] = self.country_embedding(X[:, 7].long()).squeeze(2)
        X[:, 8] = self.city_embedding(X[:, 8].long()).squeeze(2)
        X[:, 9] = self.country_embedding(X[:, 9].long()).squeeze(2)
        X[:, 10] = self.city_embedding(X[:, 10].long()).squeeze(2)
        X[:, 11] = self.country_embedding(X[:, 11].long()).squeeze(2)
        X[:, 12] = self.city_embedding(X[:, 12].long()).squeeze(2)
        X[:, 13] = self.country_embedding(X[:, 13].long()).squeeze(2)

        X[:, 14] = self.year_embedding(X[:, 14].long()).squeeze(2)
        X[:, 15] = self.month_embedding(X[:, 15].long()).squeeze(2)
        X[:, 16] = self.day_embedding(X[:, 16].long()).squeeze(2)
        X[:, 17] = self.year_embedding(X[:, 17].long()).squeeze(2)
        X[:, 18] = self.month_embedding(X[:, 18].long()).squeeze(2)
        X[:, 19] = self.day_embedding(X[:, 19].long()).squeeze(2)

        for rnn in self.rnns:
            X = rnn(X)

        X = self.flatten(X)

        X = self.mlp(X)

        X = X * self.city_embedding.weight.T + self.output_bias

        return X
