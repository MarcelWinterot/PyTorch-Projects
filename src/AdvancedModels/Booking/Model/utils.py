import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearComposition(nn.Module):
    def __init__(self):
        super(BilinearComposition, self).__init__()
        self.weights = nn.Parameter(torch.randn(11987, 11987))
        self.biases = nn.Parameter(torch.randn(11987))

        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.biases, 0.0)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, X_2):
        X = torch.matmul(X, self.weights)

        X = X * X_2

        X = X + self.biases

        X = self.softmax(X)

        return X


class MLPBlock(nn.Module):
    def __init__(self, activation, dropout: float = 0.0, use_norm: bool = False, last_layer: bool = False) -> None:
        super(MLPBlock, self).__init__()
        self.activation = activation
        self.use_norm = use_norm
        self.last_layer = last_layer

        self.linear_1 = nn.Linear(2816, 4096)
        self.linear_2 = nn.Linear(4096, 8192)
        self.linear_3 = nn.Linear(8192, 11987)

        if use_norm:
            self.norm_1 = nn.BatchNorm1d(9)
            self.norm_2 = nn.BatchNorm1d(9)
            self.norm_3 = nn.BatchNorm1d(9)

        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.tensor) -> torch.tensor:
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

        if not self.last_layer:
            X = self.activation(X)
            X = self.drop(X)

            if self.use_norm:
                X = self.norm_3(X)

        return X
