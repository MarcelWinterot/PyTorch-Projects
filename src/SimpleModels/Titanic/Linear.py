import torch.nn as nn

model = nn.Sequential()

model.add_module("Input linear", nn.Linear(10))
