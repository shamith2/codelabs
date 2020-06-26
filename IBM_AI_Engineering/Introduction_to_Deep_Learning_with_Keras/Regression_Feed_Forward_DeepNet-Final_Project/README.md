## Introduction to Deep Learning & Neural Networks with Keras

In this course project, you will build a regression model to model the data about concrete compressive strength using the deep learning Keras library, and then you will experiment with increasing the number of training epochs and changing number of hidden layers and you will see how changing these parameters impacts the performance of the model.
The predictors in the data of concrete strength include:
  -  Cement
  -  Blast Furnace Slag
  - Fly Ash
  -  Water
  -  Superplasticizer
  -  Coarse Aggregate
  - Fine Aggregate
    
**Prompt**

A. Build a baseline model (5 marks)

Use the Keras library to build a neural network with the following:

- One hidden layer of 10 nodes, and a ReLU activation function

- Use the adam optimizer and the mean squared error as the loss function.

1. Randomly split the data into a training and test sets by holding 30% of the data for testing. You can use the train_test_splithelper function from Scikit-learn.

2. Train the model on the training data using 50 epochs.

3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete strength. You can use the mean_squared_error function from Scikit-learn.

4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

5. Report the mean and the standard deviation of the mean squared errors.

> Part A Solution:
Mean: 322.0875447119485 and Std_Dev: 388.1740029638677.

B. Normalize the data (5 marks)

Repeat Part A but use a normalized version of the data. Recall that one way to normalize the data is by subtracting the mean from the individual predictors and dividing by the standard deviation.

How does the mean of the mean squared errors compare to that from Step A?

> Part B Solution:
Mean: 381.71979496103245 and Std_Dev: 141.1954818866688. The mean increased while the standard deviation decreased (from Part A)

C. Increate the number of epochs (5 marks)

Repeat Part B but use 100 epochs this time for training.

How does the mean of the mean squared errors compare to that from Step B?

> Part C Solution:
Mean: 166.5338187782956 and Std_Dev: 15.841868282210221. The mean decreased while the standard deviation decreased drastically (from Part B)

D. Increase the number of hidden layers (5 marks)

Repeat part B but use a neural network with the following instead:

- Three hidden layers, each of 10 nodes and ReLU activation function.

How does the mean of the mean squared errors compare to that from Step B?

> Part D Solution:
Mean: 124.50409908390981 and Std_Dev: 16.228760359720834. The mean decreased (more than Part C) while the standard deviation decreased drastically (from Part B)
