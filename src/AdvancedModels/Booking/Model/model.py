import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNBlock(nn.Module):
    def __init__(self, activation, in_channels, in_between_channels, out_channels, num_layers=1, bidirectional=False, dropout=0.0, use_norm=True):
        super(RNNBlock, self).__init__()
        self.activation = activation

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

        if hasattr(self, 'norm'):
            X = self.norm(X)

        return X


class MLPBlock(nn.Module):
    def __init__(self, activation, input_channels, dropout=0.0, use_norm=True):
        super(MLPBlock, self).__init__()
        input_size = input_channels * 9

        self.activation = activation

        self.linear_1 = nn.Linear(input_size, 2048)
        self.linear_2 = nn.Linear(2048, 1024)
        self.linear_3 = nn.Linear(1024, 512)
        self.linear_4 = nn.Linear(512, 256)
        self.linear_5 = nn.Linear(256, 195)

        if use_norm:
            self.norm = nn.LayerNorm(195)

        self.drop = nn.Dropout(dropout)

    def forward(self, X):
        X = self.linear_1(X)
        X = self.activation(X)
        X = self.drop(X)

        X = self.linear_2(X)
        X = self.activation(X)
        X = self.drop(X)

        X = self.linear_3(X)
        X = self.activation(X)
        X = self.drop(X)

        X = self.linear_4(X)
        X = self.activation(X)
        X = self.drop(X)

        X = self.linear_5(X)
        X = self.activation(X)
        X = self.drop(X)

        if hasattr(self, 'norm'):
            X = self.norm(X)

        return X


class BilinearComposition(nn.Module):
    def __init__(self, activation, input_channels, dropout=0.0, use_norm=True):
        super(BilinearComposition, self).__init__()
        self.mlp_1 = MLPBlock(activation, input_channels, dropout, use_norm)
        self.mlp_2 = MLPBlock(activation, input_channels, dropout, use_norm)

        self.weights = nn.Parameter(torch.randn(195, 195))
        self.biases = nn.Parameter(torch.randn(195))

        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.biases, 0.0)

    def forward(self, X):
        X_1 = self.mlp_1(X)
        X_2 = self.mlp_2(X)

        X = torch.matmul(X_1, self.weights)
        X = X * X_2
        X = X + self.biases

        X = F.softmax(X)

        return X


class CountryBlock(nn.Module):
    def __init__(self):
        super(CountryBlock, self).__init__()
        self.activation = nn.PReLU()

        self.rnn_1 = RNNBlock(self.activation, 1, 32, 64, num_layers=1, bidirectional=True,
                              dropout=0.0, use_norm=False)
        self.rnn_2 = RNNBlock(self.activation, 64, 96, 128, num_layers=1, bidirectional=True,
                              dropout=0.0, use_norm=False)
        self.rnn_3 = RNNBlock(self.activation, 128, 256, 256, num_layers=1, bidirectional=True,
                              dropout=0.0, use_norm=False)
        self.rnn_4 = RNNBlock(self.activation, 256, 256, 256, num_layers=2, bidirectional=True,
                              dropout=0.0, use_norm=False)

        self.flatten = nn.Flatten()

        # self.bilinear = BilinearComposition(
        #     F.relu, 256, dropout=0.0)
        self.linear_1 = nn.Linear(2304, 1152)
        self.linear_2 = nn.Linear(1152, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 195)

        self.relu = nn.ReLU()
        self.drop_02 = nn.Dropout(0.2)
        self.drop_05 = nn.Dropout(0.5)

        self.city_embedding = nn.Embedding(39901, 1)
        self.country_embedding = nn.Embedding(195, 1)

    def forward(self, X):
        X[:, 0] = self.city_embedding(X[:, 0].long()).squeeze(2)
        X[:, 1] = self.country_embedding(X[:, 1].long()).squeeze(2)
        X[:, 2] = self.country_embedding(X[:, 2].long()).squeeze(2)

        X = self.rnn_1(X)

        X = self.rnn_2(X)

        X = self.rnn_3(X)

        X = self.rnn_4(X)

        X = self.flatten(X)

        # X = self.bilinear(X)

        X = self.linear_1(X)
        X = self.activation(X)
        # X = self.drop_05(X)

        X = self.linear_2(X)
        X = self.activation(X)

        X = self.linear_3(X)
        X = self.activation(X)

        X = self.linear_4(X)

        X = F.softmax(X, dim=1)

        return X


class CityBlock(nn.Module):
    def __init__(self):
        super(CityBlock, self).__init__()
        self.rnn_1 = RNNBlock(F.relu, 1, 32, 64, num_layers=1, bidirectional=False,
                              dropout=0.0)
        self.rnn_2 = RNNBlock(F.relu, 64, 96, 128, num_layers=1, bidirectional=False,
                              dropout=0.0)
        self.rnn_3 = RNNBlock(F.relu, 128, 256, 256, num_layers=1, bidirectional=False,
                              dropout=0.0)
        self.rnn_4 = RNNBlock(F.relu, 256, 256, 256, num_layers=2, bidirectional=False,
                              dropout=0.0)

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(2304, 4096)
        self.linear_2 = nn.Linear(4096, 8192)
        # self.linear_3 = nn.Linear(8192, 16384)
        self.linear_4 = nn.Linear(8192, 39901)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.drop_02 = nn.Dropout(0.2)
        self.drop_05 = nn.Dropout(0.5)

        self.city_embedding = nn.Embedding(39901, 1)
        self.country_embedding = nn.Embedding(195, 1)

    def forward(self, X):
        X[:, 0] = self.city_embedding(X[:, 0].long()).squeeze(2)
        X[:, 1] = self.country_embedding(X[:, 1].long()).squeeze(2)
        X[:, 2] = self.country_embedding(X[:, 2].long()).squeeze(2)

        X = self.rnn_1(X)

        X = self.rnn_2(X)

        X = self.rnn_3(X)

        X = self.rnn_4(X)

        X = self.flatten(X)

        X = self.linear_1(X)
        X = self.relu(X)
        # X = self.drop_05(X)

        X = self.linear_2(X)
        X = self.relu(X)

        # X = self.linear_3(X)
        # X = self.relu(X)

        X = self.linear_4(X)

        X = self.softmax(X)

        return X


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.country_block = CountryBlock()
        self.city_block = CityBlock()

    def forward(self, X):
        country = self.country_block(X)

        country_argmax = torch.argmax(country, dim=1, keepdim=True)

        X = torch.cat((X, country_argmax), dim=1)

        city = self.city_block(X)

        return country, city
