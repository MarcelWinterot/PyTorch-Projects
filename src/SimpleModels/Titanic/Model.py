import torch
import torch.nn as nn
from Data import train_loader, test_loader, device
import numpy as np


class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(5, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


model = TitanicModel().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

EPOCHS = 1000

cpu = torch.device("cpu")

for epoch in range(1, EPOCHS + 1):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_losses = []
        for val_inputs, val_labels in test_loader:
            val_outputs = model(val_inputs)
            val_losses.append(loss_function(val_outputs, val_labels).to(cpu))
        if epoch % 50 == 0:
            print(f"Val_loss: {np.average(np.array(val_losses))}")
