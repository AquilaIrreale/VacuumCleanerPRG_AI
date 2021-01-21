import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import random
import gzip

labels = ['C','D','F','S','V','X']

with gzip.open("dataset/training.csv.gz", 'r') as f:
    train = np.genfromtxt(f, delimiter=',')

with gzip.open("dataset/testing.csv.gz", 'r') as f:
    test = np.genfromtxt(f, delimiter=',')

train_imgs = train[:, 1:]/255.
train_labels = train[:, 0].astype('int32')
test_imgs = test[:, 1:]/255.
test_labels = test[:, 0].astype('int32')

model = keras.Sequential() 

#model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(512, activation="relu", input_shape=(784,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(6, activation="softmax"))

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#history = model.fit(test_imgs, test_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)
history = model.fit(train_imgs, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_imgs, test_labels))

score = model.evaluate(test_imgs, test_labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig("accuracy_graphic.png")
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig("loss_graphic.png")
plt.show()
