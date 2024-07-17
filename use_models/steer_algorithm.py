import cv2
import numpy as np
import tensorflow as tf
from camera.camera import Cam
from scipy.ndimage import label
import serial
import time
c = Cam(index=[1])

ser = serial.Serial("COM7", 115200)
time.sleep(3)
def overlay_heatmap_on_image(image, heatmap, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
    
    heatmap_colored[:, :, 0] = 0  # Set blue channel to 0
    heatmap_colored[:, :, 2] = 0  # Set red channel to 0
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    heatmap_resized = cv2.resize(heatmap_colored, (image_rgb.shape[1], image_rgb.shape[0]))
    
    overlayed_image = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_resized, alpha, 0)
    
    return overlayed_image

def sigmoid_scaled(x, scale=10, midpoint=300, steepness=100):
    #return scale / (1 + np.exp(-(x - midpoint) / steepness))
    #return x/6
    return 0.08*x


model = tf.keras.models.load_model('heatmap_model_new/model_101_epochs_15200_retrained.h5')

while True:
    image = c.get_frame()
    image = c.resize_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    equalized = cv2.equalizeHist(blurred)
    image_final = equalized[tf.newaxis,:,:]

    predicted_heatmap = model.predict(image_final/255)
    image = np.squeeze(predicted_heatmap, axis=0)*255
    image[image > 3] = 255
    image[image < 3] = 0
    gray = np.array(gray)
    threshold = 200
    binary_image = gray > threshold
    labeled_array, num_features = label(binary_image)

    rectangles = []

    for feature in range(1, num_features + 1):
        slice_x, slice_y = np.where(labeled_array == feature)
        x_min, x_max = slice_x.min(), slice_x.max()
        y_min, y_max = slice_y.min(), slice_y.max()
        rectangles.append((y_min, x_min, y_max, x_max))
    
    rectangle_area = []

    lines = 0
    for x in rectangles[:]:
        area = (x[2]-x[0])*(x[3]-x[1])
        if area < 0:
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

    image_height, image_width = gray.shape
    white_points = []
    #cv2.imshow("image", image)
    #cv2.waitKey(0)

    for y in range(image_height - 1, -1, -1):
        for x in range(image_width):
            if image[y, x] > 200: 
                white_points.append((x, y))

    white_points = np.array(white_points)
    try:
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
                direction = 10
                if line_2_0 < line_1_0:
                    # Right
                    print("r_2")
                    x = line_1_0-line_2_0
                    if x > 150:
                        print("ERR")
                    else:
                        direction = 10-int(sigmoid_scaled(x))
                        print(direction)
                    if direction < 0:
                        direction = 0
                else:
                    # Left
                    print("l_2")
                    direction = 12+int(sigmoid_scaled(line_2_0-line_1_0))
                    print(direction)
                    if direction > 20:
                        direction = 20

                ser.write(str(direction).encode())
            else:
                y_values = np.arange(0, image_height)
                line1_fit = np.polyfit(white_points[:, 1], white_points[:, 0], 1)
                line1_func = np.poly1d(line1_fit)
                line1_x_values = line1_func(y_values)
                a1, b1 = line1_fit
                direction = 10
                if a1 > 0:
                    print("l_1")
                    direction = 19
                else:
                    print("r_1")
                    direction = 0
                ser.write(str(direction).encode())
    except:
        print("ERROR")
