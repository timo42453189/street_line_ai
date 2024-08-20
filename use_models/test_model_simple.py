import cv2
import numpy as np
import tensorflow as tf
from camera.camera import Cam
c = Cam(index=[1])

model = tf.keras.models.load_model('heatmap_model_new/model_10_epochs_19000_2_lr_scedule.h5')

while True:
    image = c.get_frame()
    image = c.resize_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    equalized = cv2.equalizeHist(blurred)
    image_final = equalized[tf.newaxis,:,:]
    predicted_heatmap = model.predict(image_final/255)
    image_o = np.squeeze(predicted_heatmap, axis=0)*255
    c.show_image(equalized)
    c.show_image(image_o)