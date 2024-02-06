import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
import pandas as pd
import numpy as np


train_df = pd.read_csv('./src/AdvancedModels/Booking/Dataset/train_set.csv')
test_df = pd.read_csv('./src/AdvancedModels/Booking/Dataset/test_set.csv')

df = pd.concat([train_df, test_df])

df = df.sort_values(['utrip_id', 'checkin'])


def cities_less_than_x(df, x):
    city_counts = df['city_id'].value_counts()

    less_than_x_cities = city_counts[city_counts < x]

    print(
        f"Number of cities with less than {x} visits: {len(less_than_x_cities)}, {len(less_than_x_cities) * (x - 1)}")


for i in range(1, 11):
    cities_less_than_x(df, i)


def remove_cities_less_than_x(df, x):
    city_counts = df['city_id'].value_counts()

    less_than_x_cities = city_counts[city_counts < x]

    df = df[~df['city_id'].isin(less_than_x_cities.index)]

    return df


df = remove_cities_less_than_x(df, 9)


def city_number_in_trip(df):
    df['city_number'] = df.groupby('utrip_id').cumcount() + 1

    return df


df = city_number_in_trip(df)


def double_data_by_reversing_trips(df):
    reversed_df = df.copy()
    reversed_df = reversed_df.sort_values(['utrip_id', 'checkin'])
    reversed_df['city_id'] = reversed_df.groupby(
        'utrip_id')['city_id'].transform(lambda x: x[::-1])
    reversed_df['hotel_country'] = reversed_df.groupby(
        'utrip_id')['hotel_country'].transform(lambda x: x[::-1])
    reversed_df['booker_country'] = reversed_df.groupby(
        'utrip_id')['booker_country'].transform(lambda x: x[::-1])

    reversed_df = city_number_in_trip(reversed_df)

    df = pd.concat([df, reversed_df])

    return df


df = double_data_by_reversing_trips(df)


df['next_hotel_country'] = df.groupby(
    'utrip_id')['hotel_country'].shift(-1)
df['next_city_id'] = df.groupby('utrip_id')['city_id'].shift(-1)

df = df.dropna(subset=['next_hotel_country', 'next_city_id'])

X = df[['checkin', 'checkout', 'city_id',
        'hotel_country', 'booker_country', 'affiliate_id', 'device_class', 'city_number', 'utrip_id']]
y = df[['next_hotel_country', 'next_city_id']]


def previous_cities(X, N):
    cities = []
    countries = []

    for i in range(1, N+1):
        cities.append(X.groupby('utrip_id')['city_id'].shift(i))
        cities[i-1] = cities[i-1].fillna(-1)
        X['previous_city_id_' + str(i)] = cities[i-1]

        countries.append(X.groupby('utrip_id')['hotel_country'].shift(i))
        countries[i-1] = countries[i-1].fillna(-1)
        X['previous_country_id_' + str(i)] = countries[i-1]

    return X


N = 4
X = previous_cities(X, N)

X = X.drop('utrip_id', axis=1)


def process_time(X):
    datetime = pd.to_datetime(X['checkin'])
    X['checkin_year'] = datetime.dt.year
    X['checkin_month'] = datetime.dt.month
    X['checkin_day'] = datetime.dt.day

    datetime = pd.to_datetime(X['checkout'])
    X['checkout_year'] = datetime.dt.year
    X['checkout_month'] = datetime.dt.month
    X['checkout_day'] = datetime.dt.day

    YearEncoder = LabelEncoder()

    years = pd.concat([X['checkin_year'], X['checkout_year']])
    YearEncoder.fit(years)

    X['checkin_year'] = YearEncoder.transform(X['checkin_year'])
    X['checkout_year'] = YearEncoder.transform(X['checkout_year'])

    MonthEncoder = LabelEncoder()

    months = pd.concat([X['checkin_month'], X['checkout_month']])
    MonthEncoder.fit(months)

    X['checkin_month'] = MonthEncoder.transform(X['checkin_month'])
    X['checkout_month'] = MonthEncoder.transform(X['checkout_month'])

    DayEncoder = LabelEncoder()

    days = pd.concat([X['checkin_day'], X['checkout_day']])
    DayEncoder.fit(days)

    X['checkin_day'] = DayEncoder.transform(X['checkin_day'])
    X['checkout_day'] = DayEncoder.transform(X['checkout_day'])

    X = X.drop(['checkin', 'checkout'], axis=1)

    return X


X = process_time(X)


def process_device(X):
    device_encoder = LabelEncoder()
    X['device_class'] = device_encoder.fit_transform(X['device_class'])

    print(X['device_class'].max())

    return X


X = process_device(X)


def process_countries(X, y):
    country_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                     unknown_value=-1)
    new_encoder = LabelEncoder()

    all_countries = pd.concat(
        [X['hotel_country'], X['booker_country'], y['next_hotel_country']])

    all_countries_encoded = country_encoder.fit_transform(
        all_countries.values.reshape(-1, 1))

    for i in range(1, N+1):
        X['previous_country_id_' + str(i)] = country_encoder.transform(
            X['previous_country_id_' + str(i)].values.reshape(-1, 1))

    all_countries_encoded = np.append(all_countries_encoded, -1)

    all_countries_encoded = new_encoder.fit_transform(all_countries_encoded)

    X['hotel_country'] = all_countries_encoded[:len(X)]
    X['booker_country'] = all_countries_encoded[len(X):len(X)+len(y)]
    y['next_hotel_country'] = all_countries_encoded[-len(y):]

    for i in range(1, N+1):
        X['previous_country_id_' + str(i)] = new_encoder.transform(
            X['previous_country_id_' + str(i)])

    return X, y


X, y = process_countries(X, y)


def process_cities(X, y):
    city_encoder = LabelEncoder()

    all_cities = pd.concat(
        [X['city_id'], y['next_city_id']])

    all_cities = np.append(all_cities, -1)

    city_encoder.fit(all_cities)

    X['city_id'] = city_encoder.transform(X['city_id'])
    y['next_city_id'] = city_encoder.transform(y['next_city_id'])

    for i in range(1, N+1):
        X['previous_city_id_' + str(i)] = city_encoder.transform(
            X['previous_city_id_' + str(i)])

    return X, y


X, y = process_cities(X, y)


def process_y_data(y):
    print(y['next_city_id'].max())

    y_city = y['next_city_id'].values
    y_country = y['next_hotel_country'].values

    y_country = pd.get_dummies(y_country)

    return y_city, y_country


y_city, y_country = process_y_data(y)

print(X.head())

X = torch.tensor(X.to_numpy())
y_country = torch.tensor(y_country.to_numpy())
y_city = torch.tensor(y_city)


torch.save(X, './src/AdvancedModels/Booking/Training/X.pt')
torch.save(y_country, './src/AdvancedModels/Booking/Training/y_country.pt')
torch.save(y_city, './src/AdvancedModels/Booking/Training/y_city.pt')

print(f"Data saved")
