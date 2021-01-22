import sys
import gzip
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


class LetterRecognizerNN:
    data_dir = Path(__file__).resolve().parent/"dataset"

    def __init__(self, model_path=None):
        if model_path is not None:
            self.model = keras.models.load_model(model_path)
            self.model.summary()
        else:
            self.model = keras.Sequential()
            self.model.add(layers.Dense(512, activation="relu", input_shape=(784,)))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.Dense(256, activation="relu"))
            self.model.add(layers.Dropout(0.3))
            self.model.add(layers.Dense(6, activation="softmax"))
            self.model.summary()
            self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def train(self, model_path=None, batch_size=128, epochs=15):
        with gzip.open(self.data_dir/"training.csv.gz") as f:
            train = np.genfromtxt(f, delimiter=",")

        with gzip.open(self.data_dir/"testing.csv.gz") as f:
            test = np.genfromtxt(f, delimiter=",")

        train_imgs = train[:, 1:]/255
        train_labels = train[:, 0].astype(np.uint8)
        test_imgs = test[:, 1:]/255
        test_labels = test[:, 0].astype(np.uint8)

        history = self.model.fit(train_imgs, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_imgs, test_labels))

        if model_path is not None:
            self.model.save(model_path)

        #image = cv2.imread("Untitled2.png")
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = (255 - image)/255

        score = self.model.evaluate(test_imgs, test_labels, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.savefig("accuracy_chart.png")

        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.savefig("loss_chart.png")

    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = (255-image.flatten()) / 255
        return np.argmax(self.model.predict(np.array([image])))


if __name__ == "__main__":
    program, command, *args = sys.argv
    if command == "train":
        nn = LetterRecognizerNN()
        nn.train(*args[:1])
    elif command == "predict":
        nn = LetterRecognizerNN(args[0])
        print(nn.predict(cv2.imread(args[1])))
