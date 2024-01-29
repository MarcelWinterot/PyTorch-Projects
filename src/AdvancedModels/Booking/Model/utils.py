import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearComposition(nn.Module):
    def __init__(self, activation, MLPBlock: nn.Module, dropout=0.0, use_norm=True):
        super(BilinearComposition, self).__init__()
        self.mlp_1 = MLPBlock(activation, dropout, use_norm)

        self.weights = nn.Parameter(torch.randn(11987, 11987))
        self.biases = nn.Parameter(torch.randn(11987))

        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.biases, 0.0)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, X_2):
        X_1 = self.mlp_1(X)

        X = torch.matmul(X_1, self.weights)

        X = X * X_2
        X = X + self.biases

        X = self.softmax(X)

        return X
