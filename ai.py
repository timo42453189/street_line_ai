from tensorflow_model.models import CnnModel0
from data_storing.store_data import StoreData
import os
import numpy as np
import tensorflow as tf

s = StoreData()
train_images = []
train_data = []

dir_list = os.listdir("data_storing/train_drive")

for file in dir_list:
    a, p = s.read(file)
    if p == "r":
        p = 16
    if p == "l":
        p = 5
    if p == "f":
        p = 10
    train_data.append(p)
    train_images.append(a/255)


train_images = np.array(train_images)
print(train_images.shape)

train_data = np.array(train_data)

split_index = int(len(train_data) * 0.3)

x_train = train_images[:split_index]
x_val = train_images[split_index:]

y_train = train_data[:split_index]
y_val = train_data[split_index:]



print("Length: ", len(train_images))
m = CnnModel0((80, 320, 1), 2)
model = m.cnn_model2()
model = m.compile(model)
model = m.train(model, x_train, y_train, x_val, y_val, 3000)
model.save("model.h5")

# #make prediction
# image, pot = s.read("119.h5")
# image = image/255
# image_final = image[tf.newaxis,:,:]

# predictions = model.predict(image_final)
# print(predictions)

# #show image with lines
# print((predictions[0][0][0][0], predictions[0][0][0][1]), (predictions[0][0][1][0], predictions[0][0][1][1]))
# print((predictions[0][1][0][0], predictions[0][1][0][1]), (predictions[0][1][1][0], predictions[0][1][1][1]))

# image = cv2.line(image, (int(predictions[0][0][0][0]), int(predictions[0][0][0][1])), (int(predictions[0][0][1][0]), int(predictions[0][0][1][1])), (255,255,255), 5)
# image = cv2.line(image, (int(predictions[0][1][0][0]), int(predictions[0][1][0][1])), (int(predictions[0][1][1][0]), int(predictions[0][1][1][1])), (200,200,200), 5)

# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

