import torch
import torch.nn as nn
import torch.nn.functional as F


class CountryBlock(nn.Module):
    def __init__(self):
        super(CountryBlock, self).__init__()
        self.lstm_1 = nn.LSTM(1, 32, 1, batch_first=True)
        self.lstm_2 = nn.LSTM(32, 64, 1, batch_first=True)
        self.lstm_3 = nn.LSTM(64, 96, 1, batch_first=True)

        self.conv_1 = nn.Conv1d(96, 128, 1, 1)
        self.conv_2 = nn.Conv1d(128, 64, 1, 1)
        self.conv_3 = nn.Conv1d(64, 32, 1, 1)

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(32 * 9, 256)
        self.linear_2 = nn.Linear(256, 192)

        self.relu = nn.ReLU()
        self.drop_02 = nn.Dropout(0.2)
        self.drop_05 = nn.Dropout(0.5)

    def forward(self, X):
        X, _ = self.lstm_1(X)
        X = self.drop_02(X)

        X, _ = self.lstm_2(X)
        X = self.drop_02(X)

        X, _ = self.lstm_3(X)
        X = self.drop_02(X)

        X = X.permute(0, 2, 1)

        X = self.conv_1(X)
        X = self.relu(X)
        X = self.drop_02(X)

        X = self.conv_2(X)
        X = self.relu(X)
        X = self.drop_02(X)

        X = self.conv_3(X)
        X = self.relu(X)
        X = self.drop_02(X)

        X = self.flatten(X)

        X = self.linear_1(X)
        X = self.relu(X)
        X = self.drop_05(X)
        X = self.linear_2(X)

        X = F.softmax(X)

        return X


class CityBlock(nn.Module):
    def __init__(self):
        super(CityBlock, self).__init__()
        self.lstm_1 = nn.LSTM(1, 32, 1, batch_first=True)
        self.lstm_2 = nn.LSTM(32, 64, 1, batch_first=True)
        self.lstm_3 = nn.LSTM(64, 96, 1, batch_first=True)

        self.conv_1 = nn.Conv1d(96, 128, 1, 1)
        self.conv_2 = nn.Conv1d(128, 256, 1, 1)
        self.conv_3 = nn.Conv1d(256, 512, 1, 1)

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(512 * 10, 512 * 20)
        self.linear_2 = nn.Linear(512 * 20, 37615)

        self.relu = nn.ReLU()
        self.drop_02 = nn.Dropout(0.2)
        self.drop_05 = nn.Dropout(0.5)

    def forward(self, X):
        X, _ = self.lstm_1(X)
        X = self.drop_02(X)

        X, _ = self.lstm_2(X)
        X = self.drop_02(X)

        X, _ = self.lstm_3(X)
        X = self.drop_02(X)

        X = X.permute(0, 2, 1)

        X = self.conv_1(X)
        X = self.relu(X)
        X = self.drop_02(X)

        X = self.conv_2(X)
        X = self.relu(X)
        X = self.drop_02(X)

        X = self.conv_3(X)
        X = self.relu(X)
        X = self.drop_02(X)

        X = self.flatten(X)

        X = self.linear_1(X)
        X = self.relu(X)
        X = self.drop_05(X)

        X = self.linear_2(X)

        X = F.softmax(X)

        return X


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.country_block = CountryBlock()
        self.city_block = CityBlock()

        self.country_embedding = nn.Embedding(195, 1)
        self.city_embedding = nn.Embedding(39900, 1)

    def forward(self, X):
        X[:, 0] = self.city_embedding(X[:, 0].long()).squeeze(0)
        X[:, 1] = self.country_embedding(X[:, 1].long()).squeeze(0)
        X[:, 2] = self.country_embedding(X[:, 2].long()).squeeze(0)

        country = self.country_block(X)

        country_argmax = torch.argmax(country, dim=1, keepdim=True)

        country_embeddeed = self.country_embedding(country_argmax)

        X = torch.cat((X, country_embeddeed), dim=1)

        city = self.city_block(X)

        return country, city
