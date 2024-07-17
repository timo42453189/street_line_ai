import tensorflow as tf
from tensorflow.keras import layers, models
import keras.backend as K
from tensorflow.keras import layers, models, regularizers, callbacks

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.01,
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

def create_compact_segmentation_model(input_shape):
    model = models.Sequential()
    
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))  # 1/2
    
    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))  # 1/4
    
    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))  # 1/8
    
    # Block 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))  # 1/16
    
    # Upsampling Block 1
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))  # 1/8
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Upsampling Block 2
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))  # 1/4
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Upsampling Block 3
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))  # 1/2
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Upsampling Block 4
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))  # 1/1
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Final Conv2D layer with a single output channel (1) and 'sigmoid' activation for binary segmentation
    model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))
    return model

def unet_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.3)(p1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.3)(p2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.3)(p3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.3)(p4)

    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def compile_model(model):
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        print("Using Learning Rate Sceduler")
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
def train(model, x_train, y_train, x_val, y_val, epochs):
        model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, shuffle=True, callbacks=[], epochs=epochs)
        return model
