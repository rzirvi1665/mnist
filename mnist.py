#import tf/keras

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np # helpful for seeing the data shapes
import matplotlib.pyplot as plt # visualizations of data (graphs)
import random

mnist = keras.datasets.mnist #grabbing the dataset from keras
(trainImages, trainLabels), (testImages, testLabels)= mnist.load_data() # split the data into partitions of the train and test set


n = random.randint(0, 59999)
plt.figure()
plt.title(f"The answer for the {n:,}th dataset is {trainLabels[n]}")
plt.imshow(trainImages[n], cmap="binary")
plt.colorbar()
plt.show()

# data preprocessing
# makes the data the similar shapes and sizes
# normalize the data : converting the data to values between 0 and 1
# clever optimization (compression)


# efficiency and better results
trainImages= trainImages / 255.0
testImages= testImages / 255.0


model = keras.Sequential([  # usually sequential because layers
    keras.layers.Flatten(input_shape=(28, 28)),      # input layer (1)
    keras.layers.Dense(128, activation="relu"),     # hidden layer (2)
    keras.layers.Dense(10, activation="softmax")     # output layer (3)
                          ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(trainImages, trainLabels, epochs=10)  # more epochs is not always better

model.summary()

testLoss, testAccuracy=model.evaluate(testImages, testLabels) # in reality, loss and accuracy is worse

