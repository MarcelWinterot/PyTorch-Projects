# Tests performed and their results

## Model tests

### 1. Increasing max_time_features to 512

#### Improved accuracy by 0.7% in the first 2 epochs

#### Don't know how it will affect the final accuracy but it's promising to try

```py
self.activation = nn.PReLU()
self.rnn_norm = False
self.drop = 0.0
self.max_time_features = 512
self.rnns_with_max_time_features = 2
self.bidirectional = True

batch_size = 2048
```

### 2. Doubling the data by reversing it

#### Worked very well, increasing the accuracy up to 42.35% on epoch 9

```py
self.activation = nn.PReLU()
self.rnn_norm = False
self.drop = 0.0
self.max_time_features = 256
self.rnns_with_max_time_features = 2
self.bidirectional = True

batch_size = 2048
```

## Tests I'd like to perform but don't have resourses to do so

### 1. Increasing batch_size to 4096/8192 for Model_1

### 2. Changing the model to the transformer version with batch size of 4096/8192
