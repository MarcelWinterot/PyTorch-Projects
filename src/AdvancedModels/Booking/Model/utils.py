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
    def __init__(self, activation, dropout: float = 0.0, use_norm: bool = False, last_layer: bool = False, num_channels: int = 256) -> None:
        super(MLPBlock, self).__init__()
        self.activation = activation
        self.use_norm = use_norm
        self.last_layer = last_layer
        self.output_size = 11988

        if num_channels == 256:
            self.linear_1 = nn.Linear(4096, 4096)
            self.linear_2 = nn.Linear(4096, 8192)
            self.linear_3 = nn.Linear(8192, 11988)

            self.linears = nn.ModuleList(
                [self.linear_1, self.linear_2, self.linear_3])

            if use_norm:
                self.norm_1 = nn.BatchNorm1d(4096)
                self.norm_2 = nn.BatchNorm1d(8192)
                self.norm_3 = nn.BatchNorm1d(11988)

                self.norms = nn.ModuleList(
                    [self.norm_1, self.norm_2, self.norm_3])

        elif num_channels == 512:
            self.linear_1 = nn.Linear(5632, 10000)
            self.linear_2 = nn.Linear(10000, 11988)

            self.linears = nn.ModuleList(
                [self.linear_1, self.linear_2])

            if use_norm:
                self.norm_1 = nn.BatchNorm1d(10000)
                self.norm_2 = nn.BatchNorm1d(11988)

                self.norms = nn.ModuleList(
                    [self.norm_1, self.norm_2])
        else:
            raise ValueError("Invalid num_channels")

        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.tensor) -> torch.tensor:
        for i, linear in enumerate(self.linears):
            X = linear(X)
            if i != len(self.linears) - 1:
                X = self.activation(X)
                X = self.drop(X)

                if self.use_norm:
                    X = self.norms[i](X)

        if not self.last_layer:
            X = self.activation(X)
            X = self.drop(X)

            if self.use_norm:
                X = self.norm_3(X)

        return X
