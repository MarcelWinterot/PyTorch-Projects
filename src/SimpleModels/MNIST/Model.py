from Data import train_loader, test_loader, device
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class MNISTModel(nn.Module):
    def __init__(self) -> None:
        super(MNISTModel, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, 1)
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(32, 64, 3, 1)
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(1600, 10)

    def forward(self, X):
        X = self.conv_1(X)
        X = self.pool_1(X)
        X = torch.nn.functional.relu(X)
        X = self.conv_2(X)
        X = self.pool_2(X)
        X = torch.nn.functional.relu(X)
        X = self.flatten(X)
        X = torch.nn.functional.dropout(X, 0.5)
        X = self.linear_1(X)
        X = torch.nn.functional.softmax(X, dim=1)

        return X


model = MNISTModel().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

EPOCHS = 100

output = model(train_loader.dataset[0][0].unsqueeze(0).to(device))


def train(EPOCHS: int, train_loader=train_loader, test_loader=test_loader):
    for epoch in range(1, EPOCHS + 1):
        print(f"STARTED EPOCH: {epoch}")

        train_loader = tqdm(train_loader, desc='Training')

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        model.train()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_samples += len(inputs)

            for i in range(len(outputs)):
                if torch.argmax(outputs[i]) == torch.argmax(labels[i]):
                    correct_predictions += 1

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples * 100

        print(
            f"Epoch: {epoch} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")

        test_loader = tqdm(test_loader, desc='Validation', leave=False)

        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        model.eval()

        with torch.no_grad():
            for val_inputs, val_label in test_loader:
                val_output = model(val_inputs)
                val_loss = loss_function(val_output, val_label)

                val_running_loss += val_loss.item()
                val_total_samples += len(val_inputs)

                for i in range(len(val_output)):
                    if torch.argmax(val_output[i]) == torch.argmax(val_label[i]):
                        val_correct_predictions += 1

        val_epoch_loss = val_running_loss / len(test_loader)
        val_epoch_accuracy = val_correct_predictions / val_total_samples * 100

        print(
            f"Epoch: {epoch} Loss: {val_epoch_loss:.4f} Accuracy: {val_epoch_accuracy:.2f}%")


train(EPOCHS)
