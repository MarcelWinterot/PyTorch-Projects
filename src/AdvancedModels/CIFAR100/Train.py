from Model import MarcelNet
from Data import train_loader
import torch.nn as nn
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MarcelNet().to(device)

for layer in model.modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')

# model.load_state_dict(torch.load(
#     'src/AdvancedModels/CIFAR100/models/best_model.pth'))

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.01,
                            momentum=0.9, weight_decay=5e-4, nesterov=True)

max_grad_norm = 5.0
EPOCHS = 200


def train(epochs, train_loader) -> None:
    for epoch in range(1, epochs + 1):
        print(f"STARTED EPOCH: {epoch}")

        model.train()

        train_loader = tqdm(train_loader, desc='Training')

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    nn.utils.clip_grad_norm_(param, max_grad_norm)

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

        torch.save(model.state_dict(), f"model_{epoch}.pth")


def validate(test_loader):
    model.eval()

    test_loader = tqdm(test_loader, desc='Validation', leave=False)

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            running_loss += loss.item()

            total_samples += len(inputs)

            for i in range(len(outputs)):
                if torch.argmax(outputs[i]) == torch.argmax(labels[i]):
                    correct_predictions += 1

    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = correct_predictions / total_samples * 100

    print(
        f"Validation finished: Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")


train(EPOCHS, train_loader)
