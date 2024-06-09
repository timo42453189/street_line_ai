import numpy as np
from data_storing.store_data import StoreData
import tensorflow as tf
from camera.camera import Cam
import os
import cv2
import matplotlib.pyplot as plt


s = StoreData()
c = Cam(index=[])

model = tf.keras.models.load_model('heatmap_model_new/model_100_epochs_5000_2.h5')
images = os.listdir("data_storing/train_images_2")

for i in images:
    img, pot = s.read2(i)
    image_final = img[tf.newaxis,:,:]

    predicted_heatmap = model.predict(image_final/255)
    image = np.squeeze(predicted_heatmap, axis=0)
    image[image > 0.06] = 1
    image[image < 0.06] = 0
    
    output_image = image.copy()
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    middle_row = image.shape[0] // 2
    row_to_check = middle_row - int(image.shape[0] * 0.3)

    # Ensure the row_to_check is within bounds
    row_to_check = min(row_to_check, image.shape[0] - 1)
    previous_pixel_value = image[row_to_check, 0]
    circles = []
    for col in range(image.shape[1]):
        current_pixel_value = image[row_to_check, col]
        if previous_pixel_value == 1 and current_pixel_value == 0 or previous_pixel_value == 0 and current_pixel_value == 1:
            print("circle")
            circles.append([col, row_to_check])
        previous_pixel_value = current_pixel_value

                
    print(circles)  
    print(circles[int(len(circles)/2)], circles[int(len(circles)/2) - 1])
    circles = [circles[int(len(circles)/2)], circles[int(len(circles)/2) - 1]]
    for circle in circles:      
        cv2.circle(output_image, (circle[0], circle[1]), radius=2, color=(0, 255, 0), thickness=2)

    c.show_image(output_image)