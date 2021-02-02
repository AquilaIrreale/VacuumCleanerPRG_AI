import os
import sys
import gzip
import math
import random

from io import BytesIO
from pathlib import Path
from statistics import mean
from operator import itemgetter
from itertools import product
from contextlib import suppress

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN


def parse_labels(file):
    labels = {}
    with open(file) as f:
        for line in f:
            cls, label = line.split()
            labels[int(cls)] = label
    return labels


def write_labels(file, labels):
    with open(file, "w") as f:
        for cls, label in labels.items():
            print(f"{cls} {label}", file=f)


class LetterRecognizerNN:
    def __init__(self, model_path=None):
        if model_path is not None:
            model_path = Path(model_path)
            self.labels = parse_labels(model_path/"classes")
            self.model = keras.models.load_model(model_path)
        else:
            input_cnn = layers.Input(shape=(784,))

            layer = layers.Reshape((28, 28, 1), input_shape=(784,))(input_cnn)

            # cnn layer
            cnn1 = layers.Conv2D(32, (3, 3), 1, padding='same', activation='relu')(layer)
            # cnn layer
            cnn2 = layers.Conv2D(64, (3, 3), 2, padding='same', activation='relu')(cnn1)
            poll2 = layers.MaxPooling2D((2, 2), padding='same')(cnn2)

            # inception module
            # 1 layer
            conv1_1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(poll2)
            conv1_2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(poll2)
            conv1_3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(poll2)
            pool_inception = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv1_3)
            # 2 layer
            conv2_1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(poll2)
            conv2_2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv1_1)
            conv2_3 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(conv1_2)
            conv2_4 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(pool_inception)
            # concatenate filters, assumes filters/channels last
            inception_layer = layers.concatenate([conv2_1, conv2_2, conv2_3, conv2_4], axis=-1)

            # average for spatial data, remove spatial information and put the look into the feature maps, reduce computation and overfitting
            avg = layers.GlobalAveragePooling2D()(inception_layer)

            # mlp
            dense1 = layers.Dense(128, activation="relu")(avg)
            drop1 = layers.Dropout(.5)(dense1)
            dense2 = layers.Dense(64, activation="relu")(drop1)
            drop2 = layers.Dropout(.5)(dense2)
            output = layers.Dense(6, activation="softmax")(drop2)

            self.model = Model(inputs=input_cnn, outputs=output)
            self.model.compile(
                    loss="sparse_categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])

        self.model.summary()

    def train(self, dataset_path, model_path=None, batch_size=128, epochs=15):
        dataset_path = Path(dataset_path)
        self.labels = parse_labels(dataset_path/"classes")

        with gzip.open(dataset_path/"training.csv.gz") as f:
            train = np.genfromtxt(f, delimiter=",")

        with gzip.open(dataset_path/"testing.csv.gz") as f:
            test = np.genfromtxt(f, delimiter=",")

        train_imgs = train[:, 1:]/255
        train_labels = train[:, 0].astype(np.uint8)
        test_imgs = test[:, 1:]/255
        test_labels = test[:, 0].astype(np.uint8)

        history = self.model.fit(
                train_imgs,
                train_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(test_imgs, test_labels))

        if model_path is not None:
            model_path = Path(model_path)
            self.model.save(model_path)
            write_labels(model_path/"classes", self.labels)

        score = self.model.evaluate(test_imgs, test_labels, verbose=0)

        loss = score[0]
        accuracy = score[1]

        with open("report.md", "w") as f:
            f.write("# Metriche\n")
            f.write(f"loss: {loss}\n")
            f.write(f"accuracy: {accuracy}\n\n")

        print("Test loss:", loss)
        print("Test accuracy:", accuracy)

        pred_test_labels = self.model.predict(test_imgs)
        pred_test_labels = np.argmax(pred_test_labels, axis=1)

        plt.imshow(confusion_matrix(test_labels, pred_test_labels), cmap=plt.cm.Blues)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title('Confusion matrix ')
        plt.savefig("confusion_matrix.png")
        plt.clf()

        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.savefig("accuracy_chart.png")
        plt.clf()

        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.savefig("loss_chart.png")
        plt.clf()

        try:
            keras.utils.plot_model(nn.model, to_file="model.png", show_shapes=True)
        except ImportError as e:
            print("Cannot generate graphic representation of model (model.png):")
            print("".join(e.args[0]))

    def predict(self, image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        visualize("To neural network", image)
        image = image.flatten() / 255
        return np.argmax(
                self.model.predict(
                    np.array([image])))


def adjacent_pairs(seq):
    it = iter(seq)
    first_val = next(it)
    val1 = first_val
    for val2 in it:
        yield val1, val2
        val1 = val2
    yield val1, first_val


debug_output = False
def visualize(title, image):
    if debug_output:
        cv2.imshow("visualize-window", image)
        cv2.setWindowTitle("visualize-window", title)
        while True:
            key = cv2.waitKey(0)
            if key in (ord("q"), 27):
                sys.exit(0)
            elif key == ord("\r"):
                return


def square_kern(n):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def rect_area(rect):
    x, y, w, h = rect
    return h*w


def draw_line(image, line, color, thickness=1):
    h, w, *_ = image.shape
    rho, theta = line
    a = math.cos(theta)
    b = math.sin(theta)

    if np.pi/4 <= theta < np.pi*3/4:
        p1 = 0, round(rho/b)
        p2 = w, round((rho - w*a)/b)
    else:
        p1 = round(rho/a), 0
        p2 = round((rho - h*b)/a), h

    return cv2.line(image, p1, p2, color, thickness)


def plt_fig_to_image(fig, dpi=90):
    with BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(image, 1)


def scatter_plot(title, xlabel, ylabel, *points, big=False):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for ps in points:
        ps = np.array(ps)
        xs, ys = ps.T
        size = 128 if big else 16
        ax.scatter(xs, ys, s=size, marker=".")
    visualize(title, plt_fig_to_image(fig))


def histogram(title, image, *vlines):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.hist(image.flatten(), 256, (0, 256), color="black")
    for x in vlines:
        ax.axvline(x, color="red")
    visualize(title, plt_fig_to_image(fig))


def mean_line(lines):
    return np.apply_along_axis(mean, 0, np.array(lines))


def mean_lines(lines):
    scatter_plot(
            "Raw Hough transform data",
            "Rho", "Theta", lines)
    lines = np.array(lines)
    for i, (rho, theta) in enumerate(lines):
        if theta > np.pi*3/4:
            lines[i] = -rho, theta-np.pi

    scatter_plot(
            "Rectified Hough transform data",
            "Rho", "Theta", lines)

    rhos, thetas = lines.T
    scale = abs(min(rhos)-max(rhos))/np.pi
    scaled_lines = np.column_stack((rhos, thetas*scale))

    scatter_plot(
            "Scaled rectified Hough transform data",
            "Rho", "Theta", scaled_lines)

    model = DBSCAN(eps=24, min_samples=1)
    model.fit(scaled_lines)

    clusters = {}
    for label, line in zip(model.labels_, lines):
        clusters.setdefault(label, []).append(line)

    scatter_plot(
            "DBSCAN clustered Hough transform data",
            "Rho", "Theta", *clusters.values())

    mean_lines = []
    for label, line_cluster in clusters.items():
        mean_lines.append(mean_line(line_cluster))

    scatter_plot(
            "Per-cluster mean of Hough transform data",
            "Rho", "Theta", *map(list, mean_lines), big=True)

    for i, (rho, theta) in enumerate(mean_lines):
        if theta < 0:
            mean_lines[i] = -rho, theta+np.pi

    return mean_lines


def line_intersection(l1, l2):
    rho1, theta1 = l1
    rho2, theta2 = l2
    ct1 = math.cos(theta1)
    st1 = math.sin(theta1)
    ct2 = math.cos(theta2)
    st2 = math.sin(theta2)
    det = ct1*st2 - st1*ct2
    if det:
        return (int((st2*rho1 - st1*rho2)/det),
                int((-ct2*rho1 + ct1*rho2)/det))
    else:
        raise ValueError("l1 and l2 are parallel")


def point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def distance_from_border(image, point):
    x, y = point
    h, w, *_ = image.shape
    return min(x, w-x-1, y, h-y-1)


def largest_blob(image):
    h, w = image.shape
    search_image = cv2.bitwise_and(image, 64)
    visualize("Finding largest blob", search_image)
    best_area = 0
    largest_blob = None
    for y, x in product(range(h), range(w)):
        if not search_image[y, x]:
            continue
        area, filled, _, _ = cv2.floodFill(search_image, None, (x, y), 255)
        visualize(f"Blob found at {(x, y)}", filled)
        if area > best_area:
            best_area = area
            largest_blob = np.copy(filled)
        _, search_image, _, _ = cv2.floodFill(search_image, None, (x, y), 0)
    _, largest_blob = cv2.threshold(largest_blob, 127, 255, cv2.THRESH_BINARY)
    return largest_blob


def score_blob(image, blob_color=255):
    h, w, *_ = image.shape
    return sum(
            distance_from_border(image, (x, y))
            for y, x in product(range(h), range(w))
            if image[y, x] == blob_color)


def main_blob(image):
    h, w, *_ = image.shape
    min_area = (min(h, w)//10)**2
    image = cv2.bitwise_and(image, 64)
    best_blob = None
    best_score = 0
    for y, x in product(range(h), range(w)):
        if not image[y, x]:
            continue
        area, _, _, _ = cv2.floodFill(image, None, (x, y), 255)
        if area > min_area:
            blob_score = score_blob(image, blob_color=255)
            if blob_score > best_score:
                best_score = blob_score
                _, best_blob = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        cv2.floodFill(image, None, (x, y), 0)
    return best_blob


def trim_to_content(image):
    h, w, *_ = image.shape
    x, y = w, h
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    return image[y:y+h, x:x+w]


def frame(image, relative_size):
    h, w, *_ = image.shape
    side = int(max(h, w) * relative_size)
    x = (side-w) // 2
    y = (side-h) // 2
    ret_image = np.zeros((side, side), dtype=image.dtype)
    ret_image[y:y+h, x:x+w] = image
    return ret_image


def smooth_out(image):
    for i in range(4):
        image = cv2.medianBlur(image, 3)
    image = cv2.pyrUp(image)
    image = cv2.blur(image, (3, 3))
    return image


def alpha_beta_correction(image, alpha, beta):
    image = image.astype(np.float32)
    return np.clip(image*alpha + beta, 0, 255).astype(np.uint8)


def gamma_correction(image, gamma):
    lookup_table = np.arange(256).reshape((1, 256))
    lookup_table = np.clip((lookup_table/255)**gamma * 255, 0, 255)
    lookup_table = lookup_table.astype(np.uint8)
    return cv2.LUT(image, lookup_table)


def read_board(file, model):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    visualize("Original image", image)
    histogram("Original image histogram", image)
    p01, p99 = np.quantile(image, (.01, .99))
    histogram("Original image histogram with 1st and 99th percentiles", image, p01, p99)
    alpha = 255/(p99-p01)
    beta = -p01*alpha
    corrected_image = alpha_beta_correction(image, alpha, beta)
    image = corrected_image
    histogram("Alpha-beta corrected image histogram", image)
    visualize("Alpha-beta correction", image)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    visualize("Double gaussian blur", image)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
    visualize("Adaptive threshold", image)
    image = 255-image
    visualize("Invert", image)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, square_kern(6))
    visualize("Morphological closing", image)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, square_kern(2))
    visualize("Morphological opening", image)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    figure = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    figure = cv2.drawContours(figure, contours, -1, (255, 0, 0), 3)
    visualize("External contours", figure)

    x, y, w, h = max((cv2.boundingRect(c) for c in contours), key=rect_area)
    figure = cv2.rectangle(figure, (x, y), (x+w, y+h), (0, 0, 255), 3)
    visualize("Main contour bounding box", figure)

    image = image[y:y+h, x:x+w]
    corrected_image = corrected_image[y:y+h, x:x+w]
    visualize("Crop to main contour", image)

    grid = largest_blob(image)
    visualize("Largest blob (grid)", grid)

    lines = cv2.HoughLines(grid, 1, np.pi/180, 200)
    lines = lines[:, 0, :]
    figure = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    for line in lines:
        figure = draw_line(figure, line, (0, 0, 255))
    visualize("Hough transform of grid", figure)

    lines = mean_lines(lines)
    figure = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        figure = draw_line(figure, line, (0, 0, 255), 3)
    visualize("Clusterized mean Hough transform", figure)

    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        rho, theta = line
        if np.pi/4 <= theta < np.pi*3/4:
            _, y = line_intersection(line, (w/2, 0))
            horizontal_lines.append((y, line))
        else:
            x, _ = line_intersection(line, (h/2, np.pi/2))
            vertical_lines.append((x, line))

    m = len(vertical_lines) - 1
    n = len(horizontal_lines) - 1
    print(f"Shape detected {n}x{m}")
    print()

    _, left_line   = min(vertical_lines)
    _, right_line  = max(vertical_lines)
    _, top_line    = min(horizontal_lines)
    _, bottom_line = max(horizontal_lines)

    lines = [top_line, left_line, bottom_line, right_line]
    for line in lines:
        figure = draw_line(figure, line, (255, 0, 0), 3)
    visualize("Grid edge lines", figure)

    intersections = [line_intersection(l1, l2) for l1, l2 in adjacent_pairs(lines)]
    for x, y in intersections:
        cv2.circle(figure, (x, y), 9, (0, 255, 0), -1)
    visualize("Grid corner points", figure)

    top_left, bottom_left, bottom_right, top_right = intersections
    w = int(max(point_distance(p1, p2) for p1, p2 in ((top_left, top_right), (bottom_left, bottom_right))))
    h = int(max(point_distance(p1, p2) for p1, p2 in ((top_left, bottom_left), (top_right, bottom_right))))

    src = np.array(intersections, dtype=np.float32)
    dst = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(corrected_image, transform_matrix, (w, h), flags=cv2.INTER_LINEAR)
    visualize("Perspective correction of original image", image)

    #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, square_kern(6))
    #visualize("Morphological closing", image)

    classes = np.zeros((n, m), dtype=np.uint8)

    cell_h = h//n
    cell_w = w//m
    for i, j in product(range(n), range(m)):
        cell_y = i*cell_h
        cell_x = j*cell_w
        cell = image[
                cell_y:cell_y+cell_h,
                cell_x:cell_x+cell_w]
        visualize("Grid cell", cell)
        cell = cv2.GaussianBlur(cell, (5, 5), 0)
        visualize("Gaussian blur", cell)
        value, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        visualize(f"Otsu's threshold ({value})", cell)
        cell = 255-cell
        visualize("Invert", cell)
        cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, square_kern(2))
        visualize("Morphological opening", cell)
        letter = main_blob(cell)
        figure = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
        figure[letter == 255] = (0, 0, 255)
        visualize("Main blob", figure)
        letter = trim_to_content(letter)
        visualize("Trim", letter)
        letter = frame(letter, 1.5)
        visualize("Frame", letter)
        letter = smooth_out(letter)
        visualize("Smooth out", letter)
        classes[i, j] = model.predict(letter)

    return classes


def print_board(board, labels):
    n, m = board.shape
    print(f"{'+---'*m}+")
    for row in board:
        print(f"| {' | '.join(labels[c] for c in row)} |")
        print(f"{'+---'*m}+")


if __name__ == "__main__":
    program, command, *args = sys.argv

    if command == "train":
        nn = LetterRecognizerNN()
        nn.train(args[0], *args[1:2])
        sys.exit(0)

    model_path, image_file = args
    model_path = Path(model_path)

    print("Loading model...")
    print()
    nn = LetterRecognizerNN(model_path)
    print()

    if command == "predict":
        print(nn.labels[nn.predict(cv2.imread(image_file))])

    elif command == "read-board":
        classes = read_board(image_file, nn)
        print_board(classes, nn.labels)
        print()
