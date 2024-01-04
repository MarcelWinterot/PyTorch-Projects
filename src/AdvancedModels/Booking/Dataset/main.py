"""Plan for the data:

1. Split the dataset into X and y
X: All the data about current city
y: Next city and country of the with the same utrip_id

We'd have to remove the last element of each trip, but we have enough data to do so,
if we're lacking in data we will just add test set to training

2. Process the data
- Turn checkin/out time into year, month, day
- Turn the countries into numbers and then embedding
- Drop the device_class and affiliate_id for now, as I don't see them being useful
- Remove the user_id and utrip_id
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv('./src/AdvancedModels/Booking/Dataset/train_set.csv')

train_df = train_df.sort_values(['utrip_id', 'checkin'])

train_df['next_hotel_country'] = train_df.groupby(
    'utrip_id')['hotel_country'].shift(-1)
train_df['next_city_id'] = train_df.groupby('utrip_id')['city_id'].shift(-1)

train_df = train_df.dropna(subset=['next_hotel_country', 'next_city_id'])

X = train_df[['checkin', 'checkout', 'city_id',
              'hotel_country', 'booker_country']]
y = train_df[['next_hotel_country', 'next_city_id']]


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


print(X.head())
print(y.head())
