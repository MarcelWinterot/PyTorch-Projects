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

### 3. Using transformer with batch_size 512

#### Didn't work, the accuracy decreased and training time increased to 1h/epoch

### 4. Increasing batch_size to 4096 for Model_1

#### Didn.t work, the accuracy didn't increase from 3%

### 5. Increasing number of RNN blocks and adding RNN normalization

#### Didn't work. The model's performance decreased during training

### 6. Using an embedding for each parameter

#### Similar results to not using embeddings for dates

### 7. Using AdamW optimizer with LRS

#### TODO

### 8. Decreasing the batch_size

#### TODO
