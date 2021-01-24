import math
import numpy as np
import cv2


def visualize(image):
    cv2.imshow("Vision2 image processing step", image)
    cv2.waitKey(0)


def square_se(n):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def bb_area(bb):
    x, y, w, h = bb
    return h*w


def draw_line(image, rho, theta, color):
    h, w, *_ = image.shape
    if theta:
        m = -1 / math.tan(theta);
        q = rho / math.sin(theta);
        return cv2.line(image, (0, int(q)), (w, int(m*w+q)), color)
    else:
        return cv2.line(image, (int(rho), 0), (int(rho), h), color)


image = cv2.imread("images/photo_2021-01-24_12-36-39.jpg", cv2.IMREAD_GRAYSCALE)
visualize(image)
image = cv2.GaussianBlur(image, (11, 11), 0)
visualize(image)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 4)
visualize(image)
image = 255-image
visualize(image)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, square_se(6))
visualize(image)
image = cv2.morphologyEx(image, cv2.MORPH_OPEN, square_se(3))
visualize(image)

contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
figure = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
figure = cv2.drawContours(figure, contours, -1, (255, 0, 0), 3)
visualize(figure)

x, y, w, h = max((cv2.boundingRect(c) for c in contours), key=bb_area)
figure = cv2.rectangle(figure, (x, y), (x+w, y+h), (0, 0, 255), 3)
visualize(figure)

image = image[y:y+h, x:x+w]
visualize(image)

search_image = cv2.bitwise_and(image, 64)
visualize(search_image)

best_area = 0
largest_blob = None
for y in range(h):
    for x in range(w):
        if not search_image[y, x]:
            continue
        area, filled, _, _ = cv2.floodFill(search_image, None, (x, y), 255)
        visualize(filled)
        if area > best_area:
            best_area = area
            largest_blob = np.copy(filled)
        _, search_image, _, _ = cv2.floodFill(search_image, None, (x, y), 0)

_, largest_blob = cv2.threshold(largest_blob, 127, 255, cv2.THRESH_BINARY)
visualize(largest_blob)

lines = cv2.HoughLines(largest_blob, 1, np.pi/180, 150)
figure = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for line in lines:
    rho, theta = line[0]
    figure = draw_line(figure, rho, theta, (0, 0, 255))

visualize(figure)
