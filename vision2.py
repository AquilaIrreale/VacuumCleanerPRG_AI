import sys
import math
import statistics
from operator import itemgetter

import numpy as np
import cv2
from sklearn.cluster import DBSCAN


debug_output = True
def visualize(image):
    if debug_output:
        cv2.imshow("Vision2 image processing step", image)
        cv2.waitKey(0)


def square_se(n):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def bb_area(bb):
    x, y, w, h = bb
    return h*w


def draw_line(image, line, color, thickness=1):
    rho, theta = line
    h, w, *_ = image.shape
    if theta:
        m = -1 / math.tan(theta);
        q = rho / math.sin(theta);
        return cv2.line(image, (0, int(q)), (w, int(m*w+q)), color, thickness)
    else:
        return cv2.line(image, (int(rho), 0), (int(rho), h), color, thickness)


def mean_line(lines):
    return np.array([
        statistics.mean(line[0] for line in lines),
        statistics.mean(line[1] for line in lines),
    ])


def adjacent_pairs(seq):
    it = iter(seq)
    first_val = next(it)
    val1 = first_val
    for val2 in it:
        yield val1, val2
        val1 = val2
    yield val1, first_val


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


image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
visualize(image)
image = cv2.GaussianBlur(image, (11, 11), 0)
visualize(image)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 4)
visualize(image)
image = 255-image
visualize(image)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, square_se(6))
visualize(image)
image = cv2.morphologyEx(image, cv2.MORPH_OPEN, square_se(2))
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
lines = lines[:, 0, :]
figure = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for line in lines:
    figure = draw_line(figure, line, (0, 0, 255))
visualize(figure)

cluster_model = DBSCAN(eps=8, min_samples=1)
cluster_model.fit(lines)

clusters = {}
for label, line in zip(cluster_model.labels_, lines):
    clusters.setdefault(label, []).append(line)

figure = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
mean_lines = []
for label, line_cluster in clusters.items():
    ml = mean_line(line_cluster)
    mean_lines.append(ml)
    figure = draw_line(figure, ml, (0, 0, 255), 3)
visualize(figure)

horizontal_lines = []
vertical_lines = []
for line in mean_lines:
    rho, theta = line
    if np.pi/4 <= theta < np.pi*3/4:
        horizontal_lines.append(line)
    else:
        vertical_lines.append(line)

m = len(vertical_lines) - 1
n = len(horizontal_lines) - 1
print((n, m))

left_line = max(vertical_lines, key=itemgetter(0))
right_line = min(vertical_lines, key=itemgetter(0))
top_line = min(horizontal_lines, key=itemgetter(0))
bottom_line = max(horizontal_lines, key=itemgetter(0))

lines = [top_line, left_line, bottom_line, right_line]
for line in lines:
    figure = draw_line(figure, line, (255, 0, 0), 3)
visualize(figure)

intersections = [line_intersection(l1, l2) for l1, l2 in adjacent_pairs(lines)]
for x, y in intersections:
    cv2.circle(figure, (x, y), 6, (0, 255, 0), -1)
visualize(figure)

top_left, bottom_left, bottom_right, top_right = intersections
w = int(max(point_distance(p1, p2) for p1, p2 in ((top_left, top_right), (bottom_left, bottom_right))))
h = int(max(point_distance(p1, p2) for p1, p2 in ((top_left, bottom_left), (top_right, bottom_right))))

src = np.array(intersections, dtype=np.float32)
dst = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype=np.float32)
transform_matrix = cv2.getPerspectiveTransform(src, dst)
image = cv2.warpPerspective(image, transform_matrix, (w, h), flags=cv2.INTER_NEAREST)
visualize(image)

image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, square_se(6))
visualize(image)
