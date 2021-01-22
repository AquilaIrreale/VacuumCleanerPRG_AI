import cv2
import numpy as np


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


if __name__ == "__main__":
    board = read_board('images/digital.jpg')
    print(board.shape)
    for row in board:
        for image in row:
            cv2.imshow("Test image", image)
            cv2.waitKey(0)
