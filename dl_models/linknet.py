from keras.optimizers import *
from keras.layers import *
from keras.callbacks import *
from keras.models import Model

def encoder_block(i, n, m, batch_norm=False):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    
    x = Conv2D(n, (3, 3), strides=2, padding='same')(i)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(n, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(n, (1, 1), strides=2)(i)
    x = Add()([x, shortcut])
    i = x

    x = Conv2D(n, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(n, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)

    x = Add()([x, i])
    return x
    
def decoder_block(i, n, m, batch_norm=False):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3

    x = Conv2D(m // 4, (1, 1), padding='same')(i)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(m // 4, (3, 3), strides=2, padding='same')(x)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(n, (1, 1), padding='same')(x)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)    

    return x

def get_linknet(img_shape, n_out=1, batch_norm=False, dropout=0.0, batch_norm_in=False):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3

    n = [64, 128, 256, 512]
    m = [64, 64, 128, 256]

    inputs = Input(img_shape)
   
    x = inputs
    if batch_norm_in:
        x = BatchNormalization(axis=axis)(x)

    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((3, 3), strides=2)(x)

    e1 = encoder_block(x, n[0], m[0], batch_norm)
    e2 = encoder_block(e1, n[1], m[1], batch_norm)
    e3 = encoder_block(e2, n[2], m[2], batch_norm)
    e4 = encoder_block(e3, n[3], m[3], batch_norm)

    d4 = decoder_block(e4, m[3], n[3], batch_norm)
    d4 = Add()([e3, d4])
    d3 = decoder_block(d4, m[2], n[2], batch_norm)
    d3 = Add()([e2, d3])
    d2 = decoder_block(d3, m[1], n[1], batch_norm)
    d2 = Add()([e1, d2])
    d1 = decoder_block(d2, m[0], n[0], batch_norm)

    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(d1)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)

    if dropout > 0:
        x = SpatialDropout2D(dropout)(x)

    x = Conv2DTranspose(n_out, (2, 2), strides=2, padding='same')(x)
    if batch_norm:
        x = BatchNormalization(axis=axis)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    return model
