# Plans for improvements

## Dataset

### 1. Removing cities that only show up once or twice throught the whole dataset

#### Don't know if it will work, but it's worth to give it a try

### 2. Extracting better features from the dataset

#### I belive the data we have currently are not the best, so I will have to play with the data in the near future

## Model

### 1. Experimenting with using different layers

#### Don't know if it will work, but experimenting with attention or convolutional layers is worth a shot

### 2. We can improve the MLP at the end by switching it to cpu and changing it

#### I can try to use the bidirectional composition as the MLP at the end, but the problem is memory

#### To counter this I will try moving the MLP to cpu and have the model work on 2 seperate devices

### 3. Creating a model that acts as a lookup table for cities to countries

#### Assuming current philosophy we create a model that predicts city and then a lookup table for city to country

#### We can either create it or create a model that acts as it
