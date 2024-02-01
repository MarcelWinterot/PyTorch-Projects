# Plans for the future

### 1. Test if the 30% accuracy is because of overfitting or not

#### If it is then terrible news as I have to add dropout and all the other stuff

### 2. Add an embedding to each value in the input data

#### Probably will help as adding only 2 embedding increased the accuracy in previous model from 8.0 to 8.5%

### 3. Add data.normal_() to each embedding

#### I see it all the time when initalizating embeddings so lets hope it helps

### 4. Changing the model to the transfromer (model_2) and increasing batch size to 4096 or 8192

#### I want to see the final accuracy of the model on @4 accuracy, and see if I can finally break the barrier of 60%
