from keras.optimizers import *
from keras.layers import *
from keras.callbacks import *
from keras.models import Model

# U-Net implementation
# https://arxiv.org/pdf/1505.04597.pdf

def double_conv_layer(x, size, dropout, batch_norm):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv

def get_unet(img_shape, dropout_val=0.0, batch_norm=False, init_filters=32, n_blocks=5):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3

    inputs = Input(img_shape)
    x = inputs
    convs = []

    # Contracting path
    for i in range(n_blocks):
        x = double_conv_layer(x, 2**i * init_filters, dropout_val, batch_norm)
        convs.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = double_conv_layer(x, 2**(n_blocks) * init_filters, dropout_val, batch_norm)
    convs.append(x)

    # Expansive path
    for i in range(n_blocks - 1, -1, -1):
        x = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(x), convs[i]])
        x = double_conv_layer(x, 2**i * init_filters, dropout_val, batch_norm)
    
    x = Conv2D(1, (1, 1))(x)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
