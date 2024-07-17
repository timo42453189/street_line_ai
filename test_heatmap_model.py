import cv2
import numpy as np
from data_storing.store_data import StoreData
import tensorflow as tf
from camera.camera import Cam
import os
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


model_2 = tf.keras.models.load_model('use_models/heatmap_model_new/model_15_epochs_19000.h5')
model = tf.keras.models.load_model('use_models/heatmap_model_new/model_100_epochs_5000.h5')
images = os.listdir("data_storing/train_images_2")
for i in images:
    img, pot = s.read(i)
    image_final = img[tf.newaxis,:,:]

    predicted_heatmap_1 = model.predict(image_final/255)
    predicted_heatmap_1 = np.squeeze(predicted_heatmap_1, axis=0)
    overlay_1 = overlay_heatmap_on_image(img, predicted_heatmap_1)
    cv2.imshow("old", predicted_heatmap_1)
    cv2.waitKey(0)

    predicted_heatmap_2 = model_2.predict(image_final/255)
    predicted_heatmap_2 = np.squeeze(predicted_heatmap_2, axis=0)

    overlay_2 = overlay_heatmap_on_image(img, predicted_heatmap_2)
    cv2.imshow("new", predicted_heatmap_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()