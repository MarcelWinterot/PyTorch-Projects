from Data import train_loader, test_loader, device
import torch.nn as nn
import torch
import numpy as np


class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 9),
            nn.ReLU(),
            nn.Linear(9, 3),
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)

        x = self.softmax(x)
        return x


model = IrisModel().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)


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
