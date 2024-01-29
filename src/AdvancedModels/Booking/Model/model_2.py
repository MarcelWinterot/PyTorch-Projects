import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.utils import BilinearComposition


class MLPBlock(nn.Module):
    def __init__(self, activation, dropout=0.0, use_norm=False):
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


class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.cpu = torch.device('cpu')
        self.gpu = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.activation = nn.PReLU()

        amount_of_layers = 6

        self.rnn_1 = nn.LSTM(1, 64, batch_first=True)
        self.rnn_2 = nn.LSTM(64, 128, batch_first=True)
        self.rnn_3 = nn.LSTM(128, 256, batch_first=True)

        self.rnns = nn.ModuleList(
            [self.rnn_1, self.rnn_2, self.rnn_3])

        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(
            256, 8, 2048, activation=self.activation, batch_first=True) for _ in range(amount_of_layers)])

        self.bilinear = BilinearComposition(
            self.activation, MLPBlock, use_norm=False)

        self.flatten = nn.Flatten()

        self.softmax = nn.Softmax(dim=1)

        self.city_embedding = nn.Embedding(11987, 1)
        self.country_embedding = nn.Embedding(195, 1)

    @staticmethod
    def generate_d_model_values(start=64, end=512, times: int = 6):
        d_model_values = []
        current_value = start

        for _ in range(times):
            d_model_values.append(current_value)
            if current_value < end:
                current_value *= 2

        return d_model_values

    def forward(self, X):
        cities = F.one_hot(
            X[:, 0].long(), num_classes=11987).squeeze(1).to(self.cpu)

        X[:, 0] = self.city_embedding(X[:, 0].long()).squeeze(2)
        X[:, 1] = self.country_embedding(X[:, 1].long()).squeeze(2)
        X[:, 2] = self.country_embedding(X[:, 2].long()).squeeze(2)

        for layer in self.rnns:
            X, _ = layer(X)
            X = self.activation(X)

        for layer in self.encoder_layers:
            X = layer(X)

        X = self.flatten(X)

        X = self.bilinear(X, cities.to(self.gpu))

        X = self.softmax(X)

        return X
