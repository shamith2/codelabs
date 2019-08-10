# importing necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np

# training data
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# normalizing data
training_images  = training_images / 255.0
test_images = test_images / 255.0

# model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
	       tf.keras.layers.Dense(128,activation=tf.nn.relu),
               tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# compiling model
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training model
model.fit(training_images, training_labels, epochs=5)

# testing data
model.evaluate(test_images, test_labels)

