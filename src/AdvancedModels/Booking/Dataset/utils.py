from torch.utils.data import Dataset


class PartDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FullDataset(Dataset):
    def __init__(self, X, y_city, y_country):
        self.X = X
        self.y_city = y_city
        self.y_country = y_country

    def __len__(self):
        return len(self.y_city)

    def __getitem__(self, idx):
        return self.X[idx], self.y_city[idx], self.y_country[idx]
