import numpy as np
from data_storing.store_data import StoreData
import tensorflow as tf
from camera.camera import Cam
import os
import cv2
import matplotlib.pyplot as plt
s = StoreData()

def overlay_heatmap_on_image(image, heatmap, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
    
    heatmap_colored[:, :, 0] = 0  # Set blue channel to 0
    heatmap_colored[:, :, 2] = 0  # Set red channel to 0
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    heatmap_resized = cv2.resize(heatmap_colored, (image_rgb.shape[1], image_rgb.shape[0]))
    
    overlayed_image = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_resized, alpha, 0)
    
    return overlayed_image


model = tf.keras.models.load_model('heatmap_model_new/model_101_epochs_9700.h5')
images = os.listdir("data_storing/train_images_2")
for i in images[100:]:
    img, pot = s.read(i)
    image_final = img[tf.newaxis,:,:]

    predicted_heatmap = model.predict(image_final/255)
    image = np.squeeze(predicted_heatmap, axis=0)*255
    image[image > 0.9] = 255
    image[image < 0.9] = 0
    gray = image.astype(np.uint8)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate the edges to make lines thicker
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask image
    mask = np.zeros_like(gray)

    # Fill the contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    centers_of_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_of_line = (x + w // 2, y + h // 2)
        centers_of_lines.append(center_of_line)
    if len(centers_of_lines) >= 2:
        center_of_line1 = centers_of_lines[0]
        center_of_line2 = centers_of_lines[1]
        center_of_lane = ((center_of_line1[0] + center_of_line2[0]) // 2, (center_of_line1[1] + center_of_line2[1]) // 2)
        image_center = (image.shape[1] // 2, image.shape[0])
        if center_of_lane[0] < image_center[0]:
            direction = "Left"
        elif center_of_lane[0] > image_center[0]:
            direction = "Right"
        else:
            direction = "Center"
    else:
        direction = "Not enough lines detected"

    # Display the result
    print(f"Steer to the {direction}")

    # Visualize the results
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(centers_of_lines) >= 2:
        cv2.circle(output_image, center_of_line1, 5, (255, 0, 0), -1)
        cv2.circle(output_image, center_of_line2, 5, (255, 0, 0), -1)
        cv2.circle(output_image, center_of_lane, 5, (0, 255, 0), -1)
        cv2.line(output_image, image_center, center_of_lane, (0, 255, 0), 2)
        
    plt.figure(figsize=(10, 5))
    plt.title('Lane Center Detection')
    plt.imshow(image)  # Correct BGR to RGB conversion
    plt.axis('off')
    plt.show()