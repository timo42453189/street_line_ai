import cv2
import numpy as np
from data_storing.store_data import StoreData
import tensorflow as tf
from camera.camera import Cam
import os

s = StoreData()
c = Cam(index=[0])

def overlay_heatmap_on_image(image, heatmap, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
    
    heatmap_colored[:, :, 0] = 0  # Set blue channel to 0
    heatmap_colored[:, :, 2] = 0  # Set red channel to 0
    
    image_rgb = image
    
    heatmap_resized = cv2.resize(heatmap_colored, (image_rgb.shape[1], image_rgb.shape[0]))
    
    overlayed_image = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_resized, alpha, 0)
    
    return overlayed_image


image = c.get_frame()
image = c.resize_image(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (15, 15), 0)
equalized = cv2.equalizeHist(blurred)

model = tf.keras.models.load_model('model1.h5')
image_final = equalized[tf.newaxis,:,:]

predicted_heatmap = model.predict(image_final/255)
predicted_heatmap = np.squeeze(predicted_heatmap, axis=0)
overlay = overlay_heatmap_on_image(image, predicted_heatmap)
c.show_image(equalized)