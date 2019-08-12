									   # Lab-2 : Computer Vision
# importing necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np

# callbacks
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.87):
      print("\nReached 87% accuracy so cancelling training!")
      self.model.stop_training = True

call_back = myCallback()

# training data
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# normalizing data
training_images  = training_images / 255.0
test_images = test_images / 255.0

# model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(512, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# compiling model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training model
model.fit(training_images, training_labels, epochs=5, callbacks=[call_back])

# testing data
model.evaluate(test_images, test_labels)

# exercise 1
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

