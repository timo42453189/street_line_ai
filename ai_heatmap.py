from tensorflow_model.heatmap_models import *
from data_storing.store_data import StoreData
import os
import numpy as np
import tensorflow as tf

s = StoreData()
train_images = []
train_data = []


class EarlyStoppingByLoss(tf.keras.callbacks.Callback):
    def __init__(self, monitor='loss', value=0.1, verbose=0):
        super(EarlyStoppingByLoss, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        #lr = self.model.optimizer.lr.learning_rate
        #print(f'Lernrate am Ende von Epoche {epoch}: {lr}')
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.value:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: early stopping, {self.monitor} is less than {self.value}.")
            self.model.stop_training = True

dir_list = os.listdir("data_storing/heatmap_data")[10000:]
print(len(dir_list))
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
split_index = int(len(train_data) * 0.1)

x_train = train_images[:split_index]
x_val = train_images[split_index:]

y_train = train_data[:split_index]
y_val = train_data[split_index:]



print("Length: ", len(train_images))
#model = create_segmentation_model((80, 320, 1))
model = tf.keras.models.load_model('use_models/heatmap_model_new/model_101_epochs_9700.h5')
model = compile_model(model)
early_stopping = EarlyStoppingByLoss(monitor='val_loss', value=0.055, verbose=1)
model = train(model, x_train, y_train, x_val, y_val, early_stopping, 100)
model.save("use_models/heatmap_model_new/model_101_epochs_15200_retrained.h5")