import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = 'test.jpg'
image_new = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

image_height, image_width = image_new.shape
white_points = []

# Iterate over each row from bottom to top
for y in range(image_height - 1, -1, -1):
    for x in range(image_width):
        if image_new[y, x] > 200:  # Threshold to consider as white point
            white_points.append((x, y))

# Convert list to numpy array for easier manipulation
white_points = np.array(white_points)

# Check if there are enough points for processing
if len(white_points) < 2:
    print("Not enough white points detected in the image to fit a line.")
else:
    # Separate the points into two lines based on their x-coordinates
    mid_x = image_width // 2
    line1_points = white_points[white_points[:, 0] < mid_x]
    line2_points = white_points[white_points[:, 0] >= mid_x]
    y_values = np.arange(0, image_height)
    # Fit linear functions to the points
    try:
        line1_fit = np.polyfit(line1_points[:, 1], line1_points[:, 0], 1)
        line1_func = np.poly1d(line1_fit)
        line1_x_values = line1_func(y_values)
    except:
        print("No left Line")
    try:
        line2_fit = np.polyfit(line2_points[:, 1], line2_points[:, 0], 1)
        line2_func = np.poly1d(line2_fit)
        line2_x_values = line2_func(y_values)

    except:
        print("No right Line")

    # Create functions for the lines
    

    # Generate y values for plotting

    # Generate x values using the fitted functions

    # Plot the results to ensure correctness
    plt.figure(figsize=(10, 5))
    plt.imshow(image_new, cmap='gray')
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

    line1_fit, line2_fit