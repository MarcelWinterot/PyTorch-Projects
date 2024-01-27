import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd


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


def double_data_by_reversing_trips(df):
    reversed_df = df.copy()
    reversed_df = reversed_df.sort_values(['utrip_id', 'checkin'])
    reversed_df['city_id'] = reversed_df.groupby(
        'utrip_id')['city_id'].transform(lambda x: x[::-1])
    reversed_df['hotel_country'] = reversed_df.groupby(
        'utrip_id')['hotel_country'].transform(lambda x: x[::-1])
    reversed_df['booker_country'] = reversed_df.groupby(
        'utrip_id')['booker_country'].transform(lambda x: x[::-1])

    df = pd.concat([df, reversed_df])

    return df


# df = double_data_by_reversing_trips(df)


df['next_hotel_country'] = df.groupby(
    'utrip_id')['hotel_country'].shift(-1)
df['next_city_id'] = df.groupby('utrip_id')['city_id'].shift(-1)

df = df.dropna(subset=['next_hotel_country', 'next_city_id'])

X = df[['checkin', 'checkout', 'city_id',
        'hotel_country', 'booker_country']]
y = df[['next_hotel_country', 'next_city_id']]


def process_time(X):
    datetime = pd.to_datetime(X['checkin'])
    X['checkin_year'] = datetime.dt.year
    X['checkin_month'] = datetime.dt.month
    X['checkin_day'] = datetime.dt.day

    datetime = pd.to_datetime(X['checkout'])
    X['checkout_year'] = datetime.dt.year
    X['checkout_month'] = datetime.dt.month
    X['checkout_day'] = datetime.dt.day

    YearScaler = MinMaxScaler()
    MonthScaler = MinMaxScaler()
    DayScaler = MinMaxScaler()

    X['checkin_year'] = YearScaler.fit_transform(
        X['checkin_year'].values.reshape(-1, 1))

    X['checkin_month'] = MonthScaler.fit_transform(
        X['checkin_month'].values.reshape(-1, 1))

    X['checkin_day'] = DayScaler.fit_transform(
        X['checkin_day'].values.reshape(-1, 1))

    X['checkout_year'] = YearScaler.transform(
        X['checkout_year'].values.reshape(-1, 1))

    X['checkout_month'] = MonthScaler.transform(
        X['checkout_month'].values.reshape(-1, 1))

    X['checkout_day'] = DayScaler.transform(
        X['checkout_day'].values.reshape(-1, 1))

    X = X.drop(['checkin', 'checkout'], axis=1)

    return X


X = process_time(X)


def process_countries(X, y):
    country_encoder = LabelEncoder()

    all_countries = pd.concat(
        [X['hotel_country'], X['booker_country'], y['next_hotel_country']])

    all_countries_encoded = country_encoder.fit_transform(all_countries)

    X['hotel_country'] = all_countries_encoded[:len(X)]
    X['booker_country'] = all_countries_encoded[len(X):len(X)+len(y)]
    y['next_hotel_country'] = all_countries_encoded[-len(y):]

    return X, y


X, y = process_countries(X, y)


def process_cities(X, y):
    city_encoder = LabelEncoder()

    all_cities = pd.concat([X['city_id'], y['next_city_id']])

    all_cities_encoded = city_encoder.fit_transform(all_cities)

    X['city_id'] = all_cities_encoded[:len(X)]
    y['next_city_id'] = all_cities_encoded[-len(y):]

    return X, y


X, y = process_cities(X, y)

print(X.head())


def process_y_data(y):
    print(y['next_city_id'].max())

    y_city = y['next_city_id'].values
    y_country = y['next_hotel_country'].values

    y_country = pd.get_dummies(y_country)

    return y_city, y_country


y_city, y_country = process_y_data(y)

X = torch.tensor(X.to_numpy())
y_country = torch.tensor(y_country.to_numpy())
y_city = torch.tensor(y_city)


torch.save(X, './src/AdvancedModels/Booking/Training/X.pt')
torch.save(y_country, './src/AdvancedModels/Booking/Training/y_country.pt')
torch.save(y_city, './src/AdvancedModels/Booking/Training/y_city.pt')

print(f"Data saved")
