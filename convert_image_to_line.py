from data_storing.store_data import StoreData
import tensorflow as tf
import os

s = StoreData()
model = tf.keras.models.load_model('model.h5')

images = os.listdir("data_storing/database")

for image_name in images:
    try:
        image, poti = s.read(image_name)
        image = image[84:150, :]/255
        image = image[tf.newaxis,:,:]
        predictions = model.predict(image)
        s.store_automatic(predictions[0], poti)
    except:
        print("Error in ", image_name)