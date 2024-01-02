from datasets import load_dataset
import torch.nn.functional as F
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader

dataset = load_dataset('cifar100')

train_df = dataset['train']
# test_df = dataset['test']


def process_data(df):
    transform_img = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_label = transforms.Compose([
        transforms.Lambda(lambda x: F.one_hot(
            torch.tensor(x), num_classes=100))
    ])

    X = np.ndarray(shape=(len(df), 3, 32, 32))
    y = np.ndarray(shape=(len(df), 100))

    for i in range(len(df)):
        X[i] = transform_img(df[i]['img'])
        y[i] = transform_label(df[i]['fine_label'])

    X = torch.tensor(X)

    y = torch.tensor(y)

    return (X, y)


X_train, y_train = process_data(train_df)
# X_test, y_test = process_data(test_df)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = X_train.to(device)
y_train = y_train.to(device)
# X_test = X_test.to(device)
# y_test = y_test.to(device)


class CIFAR100Dataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X.float()
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = CIFAR100Dataset(X_train, y_train)
# test_dataset = CIFAR100Dataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1000)
