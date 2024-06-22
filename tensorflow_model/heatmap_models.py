import tensorflow as tf
from tensorflow.keras import layers, models
import keras.backend as K
from tensorflow.keras import layers, models, regularizers, callbacks

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,  # Anzahl der Schritte, nach denen die Lernrate reduziert wird
    end_learning_rate=0.0001,  # Endwert der Lernrate
    power=0.5  # Power für den Polynomabfall
)


def create_segmentation_model(input_shape):
    model = models.Sequential()
    
    # Convolutional layers with Batch Normalization, L2 regularization, and Dropout
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.3))
    
    # Upsampling layers to match the desired output shape
    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Final Conv2D layer with a single output channel (1) and 'sigmoid' activation for binary heatmap
    model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))
    
    return model


def create_segmentation_model2(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same', 
                            kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                            kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                            kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', 
                            kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', 
                                     kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', 
                                     kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Final Conv2D layer with a single output channel (1) and 'sigmoid' activation for binary heatmap
    model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))
    return model
    

def compile_model(model):
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        print("Using Learning Rate Sceduler")
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
def train(model, x_train, y_train, x_val, y_val, early_stopping, epochs):
        model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, shuffle=True, callbacks=[], epochs=epochs)
        return model
