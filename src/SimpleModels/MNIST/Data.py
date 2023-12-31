from datasets import load_dataset
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

dataset = load_dataset('mnist')

train_dataset = dataset['train']
test_dataset = dataset['test']


def convert_dataset(dataset):
    X = dataset['image']
    y = dataset['label']

    X = [np.array(x) for x in X]

    X = torch.tensor(X)

    X = X / 255.0

    X = X.reshape(-1, 1, 28, 28)

    y = torch.nn.functional.one_hot(torch.tensor(y), num_classes=10)

    return X, y


X_train, y_train = convert_dataset(train_dataset)
X_test, y_test = convert_dataset(test_dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)


class MNISTDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = MNISTDataset(X_train, y_train)
test_dataset = MNISTDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)
