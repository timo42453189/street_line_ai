from tensorflow_model.model_fourier import *
from data_storing.store_data import StoreData
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

s = StoreData()
train_images = []
train_data = []

dir_list = os.listdir("data_storing/heatmap_data")

def preprocess_image(image):
    # Normalisieren Sie das Bild auf Werte zwischen 0 und 1
    image = image / 255.0
    # Wenden Sie die Fourier-Transformation an
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift)+1)
    return magnitude_spectrum

for file in dir_list:
    a, p = s.read(file)
    a = preprocess_image(a)
    p = p[:, :, np.newaxis]
    train_data.append(p)
    a = a[:, :, np.newaxis]
    train_images.append(a)


train_images = np.array(train_images)
print(train_images.shape)

train_data = np.array(train_data)
print(train_data.shape)
split_index = int(len(train_data) * 0.3)


x_train = train_images[:split_index]
x_val = train_images[split_index:]

y_train = train_data[:split_index]
y_val = train_data[split_index:]


print("Length: ", len(train_images))
model = get_model()
model = compile_model(model)
model, model_training_data = train(model, x_train, y_train, x_val, y_val, 100)

#model.save("heatmap_model_new/model_100_epochs_5000_2.h5")
image = train_images[0]
image = np.expand_dims(image, axis=0)
print(image.shape)
pred = model.predict(image)
pred = pred[0, :, :, 0]
plt.imshow(pred, cmap='gray')
plt.title('Predicted Heatmap')
plt.show()