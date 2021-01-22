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


CONTOUR_MAX_SIZE = 1000
CONTOUR_MAX_SIZE_2 = 50000


def read_board(file):
    # Load image, grayscale, and adaptive threshold
    image = cv2.imread(file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)

    # Filter out all numbers and noise to isolate only boxes
    cnts, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(
            thresh_image,
            [c for c in cnts if cv2.contourArea(c) < CONTOUR_MAX_SIZE],
            -1, (0, 0, 0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

    # Sort by top to bottom and each row by left to right
    invert = 255 - thresh_image

    cnts, _ = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # https://answers.opencv.org/question/179510/how-can-i-sort-the-contours-from-left-to-right-and-top-to-bottom/
    # TODO: improve
    def contours_key(c):
        x, y, w, h = cv2.boundingRect(c)
        imw, imh, depth = image.shape
        return x**(3/2) + y*imw

    cnts = np.array([c for c in sorted(cnts, key=contours_key)
                        if cv2.contourArea(c) < CONTOUR_MAX_SIZE_2])

    rightmost_x = 0
    for i, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        if x < rightmost_x:
            break
        rightmost_x = x
    ncols = i

    # Find bounding box and extract ROI
    def trim_by_contour(c):
        x, y, w, h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w]

    cnts = np.array([trim_by_contour(c) for c in cnts])
    return cnts.reshape((len(cnts)//i, i, *cnts.shape[1:]))


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
        breakpoint()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = (255-image.flatten()) / 255
        return np.argmax(self.model.predict(np.array([image])))


predict_board_nn = None
def predict_board(board_image):
    global predict_board_nn
    if predict_board_nn is None:
        predict_board_nn = LetterRecognizerNN("model")

    board = read_board(board_image)

    return np.array([
        np.array([
            predict_board_nn.predict(image)
            for image in row])
        for row in board])


if __name__ == "__main__":
    program, command, *args = sys.argv
    if command == "train":
        nn = LetterRecognizerNN()
        nn.train(*args[:1])
    elif command == "predict":
        nn = LetterRecognizerNN(args[0])
        print(nn.predict(cv2.imread(args[1])))
    elif command == "predict-board":
        board = predict_board(args[0])
        for row in board:
            print(" ".join(map(str, row)))
