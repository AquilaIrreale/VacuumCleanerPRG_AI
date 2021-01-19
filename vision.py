import cv2
from imutils import contours
import numpy as np


def read_board(path: str):
    # Load image, grayscale, and adaptive threshold
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,57,5)

    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    horizontal_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

    # Sort by top to bottom and each row by left to right
    invert = 255 - thresh
    cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # https://answers.opencv.org/question/179510/how-can-i-sort-the-contours-from-left-to-right-and-top-to-bottom/
    cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1] )

    sudoku_rows = []
    row = []
    for (_, c) in enumerate(cnts, 1):
        area = cv2.contourArea(c)
        if area < 50000: # controllo necessario perche senno verranno presi anche i quadrati rappresentanti l'intera griglia e i sotto quadrati
            (cnts, _) = contours.sort_contours(list([c]), method="left-to-right")
            sudoku_rows.append(cnts)

    # Find bounding box and extract ROI
    for row in sudoku_rows:
        for c in row:
            x,y,w,h = cv2.boundingRect(c)
            ROI = image[y:y+h, x:x+w]
            ROI = cv2.resize(ROI, (28, 28))
            cv2.imshow('image',ROI)
            cv2.waitKey(0)

if __name__ == "__main__":
    read_board('images/handwrite.jpg')