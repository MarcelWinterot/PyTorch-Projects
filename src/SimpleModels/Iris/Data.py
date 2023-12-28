import torch
from torch.utils.data import Dataset, DataLoader
import torch
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

df = sns.load_dataset('iris')

y = df.pop('species')
X = df


def one_hot(X: np.ndarray) -> np.ndarray:
    assert len(X.shape) == 1, f"Shape must be 1D, received: {len(X.shape)}"
    max_value = np.max(X)
    min_value = np.min(X)

    new_arr = np.zeros((X.shape[0], max_value - min_value + 1))

    for index, item in enumerate(X):
        new_arr[index, item-min_value] = 1

    return new_arr


def process_data(X: pd.DataFrame, y: pd.DataFrame):
    scaller = MinMaxScaler()

    X = scaller.fit_transform(X)

    encoder = {"setosa": 0, "versicolor": 1, "virginica": 2}

    y = y.map(encoder)

    y = one_hot(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    return convert_to_torch(X_train, X_test, y_train, y_test)


def convert_to_torch(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple:
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    return (X_train, X_test, y_train, y_test)


X_train, X_test, y_train, y_test = process_data(X, y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)


class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = IrisDataset(X_train, y_train)
test_dataset = IrisDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
