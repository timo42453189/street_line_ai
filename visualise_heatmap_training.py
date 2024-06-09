from tensorflow_model.heatmap_models import *
from data_storing.store_data import StoreData
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
s = StoreData()
train_images = []
train_data = []

dir_list = os.listdir("data_storing/heatmap_data")

for file in dir_list:
    a, p = s.read(file)
    p = p[:, :, np.newaxis]
    train_data.append(p)
    a = a[:, :, np.newaxis]
    train_images.append(a/255)


train_images = np.array(train_images)
print(train_images.shape)

train_data = np.array(train_data)
print(train_data.shape)
split_index = int(len(train_data) * 0.3)

x_train = train_images[:split_index]
x_val = train_images[split_index:]

y_train = train_data[:split_index]
y_val = train_data[split_index:]


image, x = s.read2("17.h5")
image_final = image[tf.newaxis,:,:,tf.newaxis]/255

print("Length: ", len(train_images))
print(image_final.shape)
model = create_segmentation_model((80, 320, 1))
model = compile_model(model)

predicted_heatmaps = []

epochs = 40
model = train(model, x_train, y_train, x_val, y_val, 10)

for i in range(epochs):
    print("Epoch: ", i)
    model = train(model, x_train, y_train, x_val, y_val, 1)
    predicted_heatmap = model.predict(image_final)
    predicted_heatmap = np.squeeze(predicted_heatmap, axis=0)
    predicted_heatmaps.append(predicted_heatmap)


# Visualize all heatmaps in a grid
num_heatmaps = len(predicted_heatmaps)
num_cols = 10
num_rows = (num_heatmaps // num_cols) + 1

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 2 * num_rows))

# Ensure axes is a 2D array
axes = np.array(axes).reshape(num_rows, num_cols)

for i, heatmap in enumerate(predicted_heatmaps):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(heatmap, cmap='hot', interpolation='nearest')
    axes[row, col].set_title(f"Epoch {i + 1}")
    axes[row, col].axis('off')

# Hide any unused subplots
for j in range(i + 1, num_rows * num_cols):
    row = j // num_cols
    col = j % num_cols
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()