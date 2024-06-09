import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils import get_custom_objects
import keras.backend as K
from tensorflow.keras import layers, models, regularizers, callbacks
class CnnModel0:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def steering_loss(self, y_true, y_pred):
        y_pred = tf.reduce_mean(y_pred)
        y_true = tf.reduce_mean(y_true)
        y_pred = float(y_pred)
        y_true = float(y_true)
        if y_true >= 10 and y_pred >= 10:
            diff = abs(y_true - y_pred) - 4.0
            return diff
        if y_true <= 10 and y_pred <= 10:
            diff = abs(y_true - y_pred) - 4.0
            return diff
        if y_true <= 10 and y_pred >= 10:
            diff = abs(y_true - y_pred) + 60.0
            return diff
        if y_true >= 10 and y_pred <= 10:
            diff = abs(y_true - y_pred) + 60.0
            return diff
        else:
            return 100.0
        
    def cnn_model0(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model
    
    
    def cnn_model1(self):
        model = models.Sequential()
        model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=self.input_shape))
        model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
        model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(self.num_classes))
        return model
    
    def cnn_model2(self):
        model = models.Sequential([
            layers.Conv2D(24, (5, 5), strides=(2, 2), input_shape=self.input_shape, activation='elu'),
            layers.BatchNormalization(),
            layers.Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
            layers.BatchNormalization(),
            layers.Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='elu'),
            layers.Conv2D(64, (3, 3), activation='elu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(1, activation="elu")
        ])
        return model
    
    
    def compile(self, model):
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt,
                      loss="mse",
                      metrics=["mse"])
        return model
    
    def train(self, model, x_train, y_train, x_val, y_val, epochs):
        model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=epochs, shuffle=True)
        return model