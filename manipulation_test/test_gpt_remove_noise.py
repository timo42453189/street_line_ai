from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
from scipy.ndimage import label
import cv2
import matplotlib.pyplot as plt

image_path = 'test.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_before = image.copy()

image_np = np.array(image)

threshold = 200
binary_image = image_np > threshold

labeled_array, num_features = label(binary_image)

rectangles = []

def sigmoid_scaled(x, scale=10, midpoint=300, steepness=100):
    return scale / (1 + np.exp(-(x - midpoint) / steepness))

for feature in range(1, num_features + 1):
    slice_x, slice_y = np.where(labeled_array == feature)
    x_min, x_max = slice_x.min(), slice_x.max()
    y_min, y_max = slice_y.min(), slice_y.max()
    rectangles.append((y_min, x_min, y_max, x_max))

    rectangle_area = []

lines = 0
for x in rectangles[:]:
    area = (x[2]-x[0])*(x[3]-x[1])
    if area < 600:
        print("removed")
        image = cv2.rectangle(image, (x[0], x[1]), (x[2], x[3]), color=(0, 0, 0), thickness=-1)
        rectangles.remove(x)
    else:
        lines+=1
        rectangle_area.append(area)

print(len(rectangles))
while(len(rectangles)>2):
    lowest_index = np.argmin(rectangle_area)
    image = cv2.rectangle(image, (rectangles[lowest_index][0], rectangles[lowest_index][1]), (rectangles[lowest_index][2], rectangles[lowest_index][3]), color=(0, 0, 0), thickness=-1)
    rectangle_area.pop(lowest_index)
    rectangles.pop(lowest_index)

image_height, image_width = image.shape
white_points = []

# Iterate over each row from bottom to top
for y in range(image_height - 1, -1, -1):
    for x in range(image_width):
        if image[y, x] > 200:  # Threshold to consider as white point
            white_points.append((x, y))

# Convert list to numpy array for easier manipulation
white_points = np.array(white_points)

# Check if there are enough points for processing
if len(white_points) < 2:
    print("Not enough white points detected in the image to fit a line.")
else:
    if lines > 1:
        mid_x = rectangles[0][2]
        line1_points = white_points[white_points[:, 0] < mid_x]
        line2_points = white_points[white_points[:, 0] >= mid_x]
        y_values = np.arange(0, image_height)
        line1_fit = np.polyfit(line1_points[:, 1], line1_points[:, 0], 1)
        line1_func = np.poly1d(line1_fit)
        line1_x_values = line1_func(y_values)
        line2_fit = np.polyfit(line2_points[:, 1], line2_points[:, 0], 1)
        line2_func = np.poly1d(line2_fit)
        line2_x_values = line2_func(y_values)
        a1, b1 = line1_fit
        a2, b2 = line2_fit
        y_intersect = (b2 - b1) / (a1 - a2)
        x_intersect = a1 * y_intersect + b1
        line_1_0 = (0-b1)/a1
        line_2_0 = (image_width-b2)/a2
        if line_2_0 < line_1_0:
            print(10-sigmoid_scaled(line_1_0-line_2_0))
            print("Steer Right")
        else:
            print(10+sigmoid_scaled(line_2_0-line_1_0))
            print("Steer Left")

    else:
        y_values = np.arange(0, image_height)
        line1_fit = np.polyfit(white_points[:, 1], white_points[:, 0], 1)
        line1_func = np.poly1d(line1_fit)
        line1_x_values = line1_func(y_values)


    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap='gray')
    try:
        plt.plot(line1_x_values, y_values, color='red', label='Line 1')
    except:
        pass
    try:
        plt.plot(line2_x_values, y_values, color='blue', label='Line 2')
    except:
        pass
    plt.legend()
    plt.show()
