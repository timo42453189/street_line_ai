import cv2
import numpy as np
from data_storing.store_data import StoreData
import tensorflow as tf
from camera.camera import Cam
import os
s = StoreData()

model = tf.keras.models.load_model('heatmap_model_new/model_100_epochs_5000_2.h5')
images = os.listdir("data_storing/train_images_2")
start_file = "1869.h5"
index = images.index(start_file)
images = images[index:]
images_left = len(images)
for i in images:
    img, pot = s.read2(i)
    print(i)
    image_final = img[tf.newaxis,:,:]
    predicted_heatmap = model.predict(image_final/255)
    predicted_heatmap = np.squeeze(predicted_heatmap, axis=0)
    cv2.imshow("image", predicted_heatmap)
    images_left -= 1
    print(images_left)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("w"):
        print("w")
        s.store_automatic(predicted_heatmap, "f")
    elif key == ord("a"):
        print("a")
        s.store_automatic(predicted_heatmap, "l")
    elif key == ord("d"):
        print("d")
        s.store_automatic(predicted_heatmap, "r")
    elif key == ord("s"):
        pass
