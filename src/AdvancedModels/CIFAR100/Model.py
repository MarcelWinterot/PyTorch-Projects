import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ConvolutionalLayer(nn.Module):
    def __init__(self, channels: list[int], kernel_size: int) -> None:
        """Initalizer for the parralel convolutional layer of MarcelNet

        Args:
            channels (list[int]): Amount of channels in each layer
            kernel_size (int): Kernel size of each layer
        """
        super().__init__()
        amount = len(channels) - 1

        self.convs = nn.ModuleList([nn.Conv2d(
            channels[i], channels[i+1], kernel_size, 1, 'same') for i in range(amount)])

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, X):
        for conv in self.convs:
            X = conv(X)
            X = F.relu(X)
            X = F.dropout(X, 0.1)

        X = self.pool(X)

        X = F.dropout(X, 0.25)

        return X


class MarcelNet(nn.Module):
    """
    Model architecture:

    Input -> [ Block ] * 5 -> Linear(4096, 1024) -> Linear(1024, 256) -> Linear(256, 100)

    Block architecture:
    4 parallel layers -> Concatenate the outputs -> Conv(1024, 256)

    We use the last conv layer to reduce the number of channels back to the 
    number of channels in a parallel layer.

    Parallel layer:
    [ Conv2D -> ReLU -> Dropout(0.1) ] * 4 -> MaxPool2D -> Dropout(0.25)

    Design inspired by:
    GoogLe Net - https://arxiv.org/abs/1409.4842
    Transformer (First idea was to proceed with encoder decoder architecture) - https://arxiv.org/abs/1706.03762

    """

    def __init__(self) -> None:
        super().__init__()
        number_of_blocks = 5  # Number of blocks
        starting_channels = 3  # Starting channels
        # By how much we increase the channels on each layer. multiplier * 2^layer_index
        current_multiplier = 16
        max_channels = 256  # Maximum number of channels
        number_of_parralel_layers = 4  # Number of parallel layers
        number_of_conv_layers = 4  # Number of conv layers in each parallel layer

        kernel_sizes = [1 + 2 * i for i in range(number_of_conv_layers)]
        channels = self.initalizer_channels(
            number_of_conv_layers, starting_channels, current_multiplier, max_channels, number_of_blocks)

        self.concat_layers = nn.ModuleList()

        self.conv_layers = nn.ModuleList()

        for i in range(number_of_blocks):
            self.conv_layers.append(self.initalize_layer(
                number_of_parralel_layers, channels[i], kernel_sizes))

            self.concat_layers.append(
                nn.Conv2d(channels[i][-1] * 4,  channels[i][-1], 2, 1, 'same'))

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(4096, 1024)
        self.linear_2 = nn.Linear(1024, 256)
        self.linear_3 = nn.Linear(256, 100)

    def initalizer_channels(self, number_of_layers: int, starting_channels: int, current_multiplier: int, max_channels: int, number_of_blocks: int) -> list:
        """Initalizer for the channels in the layer

        Args:
            number_of_layers (int): Number of parallel layers in each block
            starting_channels (int): Starting amount of channels
            current_multiplier (int): Multiplier for the channels
            max_channels (int): Maximum amount of channels
            number_of_blocks (int): Number of blocks

        Returns:
            list: All the channels
        """
        channels = [[] for _ in range(number_of_blocks)]

        for i in range(number_of_blocks):
            temp = [starting_channels] * (number_of_layers + 1)

            for j in range(number_of_layers):
                temp[j+1] = min(current_multiplier, max_channels)

                current_multiplier *= 2

            channels[i] = temp

            starting_channels = channels[i][-1]

        return channels

    def initalize_layer(self, amount, channels: list, kernel_size: list) -> nn.ModuleList:
        """Initalizer for a layer segment

        Args:
            amount (int): Amount of layers
            channels (list[int]): All the channels in the layer
            kernel_size (list[int]): kernel_size for each layer

        Returns:
            nn.ModuleList: All the layers
        """
        return nn.ModuleList([ConvolutionalLayer(channels, kernel_size[i]) for i in range(amount)])

    def forward(self, X):
        X = X.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)

        for i, conv_layer in enumerate(self.conv_layers):
            X = torch.cat([layer(X) for layer in conv_layer], dim=1)

            X = self.concat_layers[i](X)

        X = self.flatten(X)

        # Linear layers
        X = self.linear_1(X)
        X = F.relu(X)

        X = self.linear_2(X)
        X = F.relu(X)

        X = self.linear_3(X)

        X = F.softmax(X)

        return X


if __name__ == "__main__":
    model = MarcelNet()

    summary(model, (3, 32, 32))
