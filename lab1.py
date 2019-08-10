# importing necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras

# CREATING A NEURAL NETWORK
# create model with 1 layer, 1 neuron and input shape of 1 value
model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])

# compiling the model with 'mean squared error' as loss function
# and 'stochastic gradient descent' as optimizer
model.compile(loss='mean_squared_error',optimizer='sgd')

# TRAINING THE NEURAL NETWORK
# training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# training the neural network
model.fit(xs, ys, epochs=500)

# print result
print(model.predict([2341.0]))
