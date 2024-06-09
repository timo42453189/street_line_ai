import tensorflow as tf
from tensorflow.keras import layers


def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = layers.concatenate([x, concat_tensor])
    x = conv_block(x, num_filters)
    return x

def unet_custom_output(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    e1, p1 = encoder_block(inputs, 32)
    e2, p2 = encoder_block(p1, 64)
    e3, p3 = encoder_block(p2, 128)
    e4, p4 = encoder_block(p3, 256)
    
    # Bottleneck
    b = conv_block(p4, 512)
    
    # Decoder
    d4 = decoder_block(b, e4, 256)
    d3 = decoder_block(d4, e3, 128)
    d2 = decoder_block(d3, e2, 64)
    d1 = decoder_block(d2, e1, 32)
    
    # Output
    final_conv = layers.Conv2D(8, (3, 3), padding='same')(d1)
    output = layers.GlobalAveragePooling2D()(final_conv)
    output = layers.Reshape((2, 2, 2))(output)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def compile(model):
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=opt,
                      loss="mse",
                      metrics=["mse"])
        return model
    
def train(model, x_train, y_train, x_val, y_val, epochs):
        model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=epochs,)
        return model