import numpy as np
import cv2
from data_storing.store_data import StoreData
import os
s = StoreData()

# Load the image

images = os.listdir("data_storing/train_images_2")

def create_heatmap(image_shape, lines):
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)  # Create an empty heatmap
    for line in lines:
        (x1, y1), (x2, y2) = line
        cv2.line(heatmap, (x1, y1), (x2, y2), 1, thickness=2)  # Draw lines on the heatmap
    return heatmap
y=0
for i in images:
    image, lines = s.read(i)
    heatmap = create_heatmap(image.shape, lines)
    heatmap = heatmap / np.max(heatmap)
    s.store_automatic(image, heatmap)
    print(y)
    y+=1