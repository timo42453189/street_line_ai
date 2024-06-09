from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf
def get_model():
    model = models.Sequential()
    
    # Convolutional layers with Batch Normalization, L2 regularization, and Dropout
    model.add(layers.Conv2D(32, (3, 3), activation='elu', input_shape=(80, 320, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(64, (3, 3), activation='elu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (3, 3), activation='elu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(256, (3, 3), activation='elu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))
    
    # Upsampling layers to match the desired output shape
    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    
    # Final Conv2D layer with a single output channel (1) and 'sigmoid' activation for binary heatmap
    model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))
    
    return model


def compile_model(model):
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['mse'])
        return model
    
def train(model, x_train, y_train, x_val, y_val, epochs):
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, shuffle=True, callbacks=[], epochs=epochs)
        return model, history
