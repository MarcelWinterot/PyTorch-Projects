import os
import torch
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))

X = torch.load(os.path.join(script_dir, 'X.pt'))


y_city = torch.load(os.path.join(script_dir, 'y_city.pt'))
y_country = torch.load(os.path.join(script_dir, 'y_country.pt'))


def one_hot_to_number(y):
    return torch.argmax(y, dim=1)


y_country = one_hot_to_number(y_country.long())


def create_table(X, y_city, y_country):
    X_table = dict(zip(X[:, 0].tolist(), X[:, 1].tolist()))
    y_table = dict(zip(y_city.tolist(), y_country.tolist()))

    X_table.update(y_table)

    return X_table


table = create_table(X, y_city, y_country)

with open(os.path.join(script_dir, 'table.pkl'), 'wb') as f:
    pickle.dump(table, f)
