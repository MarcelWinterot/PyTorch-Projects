import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.utils import BilinearComposition, MLPBlock


class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.cpu = torch.device('cpu')
        self.gpu = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.activation = nn.PReLU()

        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.d_model = 256
        self.n_heads = 8
        self.d_ff = 2048

        self.rnn_1 = nn.LSTM(1, 64, batch_first=True)
        self.rnn_2 = nn.LSTM(64, 128, batch_first=True)
        self.rnn_3 = nn.LSTM(128, 256, batch_first=True)

        self.rnns = nn.ModuleList(
            [self.rnn_1, self.rnn_2, self.rnn_3])

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, self.n_heads, self.d_ff, activation=self.activation, batch_first=True), self.num_encoder_layers)

        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            self.d_model, self.n_heads, self.d_ff, activation=self.activation, batch_first=True), self.num_decoder_layers)

        self.mlp = MLPBlock(self.activation, dropout=0.0,
                            use_norm=False, last_layer=False)
        self.bilinear = BilinearComposition()

        self.flatten = nn.Flatten()

        self.softmax = nn.Softmax(dim=1)

        self.city_embedding = nn.Embedding(11988, 1)
        self.country_embedding = nn.Embedding(195, 1)
        self.affiliate_embedding = nn.Embedding(10698, 1)
        self.device_embedding = nn.Embedding(3, 1)

        self.output_bias = nn.Parameter(torch.zeros(11988))
        self.output_bias.data.normal_(0, 0.01)

    def forward(self, X):
        X[:, 0] = self.city_embedding(X[:, 0].long()).squeeze(2)
        X[:, 1] = self.country_embedding(X[:, 1].long()).squeeze(2)
        X[:, 2] = self.country_embedding(X[:, 2].long()).squeeze(2)
        X[:, 3] = self.affiliate_embedding(X[:, 3].long()).squeeze(2)
        X[:, 4] = self.device_embedding(X[:, 4].long()).squeeze(2)

        for layer in self.rnns:
            X, _ = layer(X)
            X = self.activation(X)

        encoder_out = self.encoder(X)

        X = self.decoder(X, encoder_out)

        X = self.flatten(X)

        X = self.mlp(X)

        X = X * self.city_embedding.weight.T + self.output_bias

        return X
