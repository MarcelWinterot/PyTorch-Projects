import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_data() -> tuple:
    X = sns.load_dataset("titanic")

    X = drop_data(X, ["fare", "embarked", "adult_male", "who",
                  "embark_town", "class", "alive", "deck", "alone"])

    y = X.pop("survived")

    encoder = {"male": 0, "female": 1}
    X["sex"] = X["sex"].map(encoder)

    X = X.dropna()
    y = y.loc[X.index]

    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train = np.array([[y] for y in y_train])
    y_test = np.array([[y] for y in y_test])

    return torch.from_numpy(X_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(y_train).float(), torch.from_numpy(y_test).float()


def drop_data(data: pd.DataFrame, data_to_drop: list[str]) -> pd.DataFrame:
    for column in data_to_drop:
        data.drop(column, axis=1, inplace=True)
    return data


X_train, X_test, y_train, y_test = load_data()

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
