import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, \
    Conv2DTranspose
from tensorflow.python.keras.layers import concatenate


def kaggle_u_net_direct(im_height, im_width, im_chan):
    """
    This is a kaggle recommended u-net architecture from kaggle kernels.
    https://www.kaggle.com/jesperdramsch/intro-to-seismic-salt-and-how-to-geophysics
    :return:
    """
    inputs = Input((im_height, im_width, im_chan))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return inputs, outputs


if __name__ == '__main__':
    inputs, outputs = kaggle_u_net_direct(128, 128, 1)
    kaggle_model = Model(inputs=[inputs], outputs=[outputs])
    kaggle_model.compile(optimizer='adam', loss='binary_crossentropy')
    kaggle_model.summary()

    img = np.zeros((101, 101, 1))
    label = np.zeros((101, 101, 1))
    tf.image.resize_image_with_crop_or_pad(img, 128, 128)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _img = tf.image.resize_image_with_crop_or_pad(img, 128, 128)
        _img = tf.reshape(_img, (1, 128, 128, 1))
        _label = tf.image.resize_image_with_crop_or_pad(label, 128, 128)
        _label = tf.reshape(_label, (1, 128, 128, 1))
        kaggle_model.fit(_img, _label, steps_per_epoch=1, epochs=10)
