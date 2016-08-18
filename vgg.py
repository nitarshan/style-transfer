# -*- coding: utf-8 -*-
""" nitarshan/style-transfer/vgg

VGG Imagenet model.

# References:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Trained image classification models for Keras](https://github.com/fchollet/deep-learning-models)

"""
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils.data_utils import get_file


WEIGHTS_FILE = 'vgg{}_weights_th_dim_ordering_th_kernels_notop.h5'
WEIGHTS_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/' + WEIGHTS_FILE
PIXEL_MEANS = {'R': 103.939, 'G': 116.779, 'B': 123.68}


def build_model(depth=19,pooling="max"):
    assert depth in {16,19}
    assert pooling in {"max","avg"}

    if pooling == "max":
        pooling = MaxPooling2D
    elif pooling == "avg":
        pooling = AveragePooling2D

    # Image input to model
    image = Input(shape=(3, None, None))

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_1')(image)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_2')(x)
    x = pooling((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_2')(x)
    x = pooling((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_3')(x)
    if depth == 19:
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_4')(x)
    x = pooling((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_3')(x)
    if depth == 19:
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_4')(x)
    x = pooling((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_3')(x)
    if depth == 19:
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_4')(x)
    x = pooling((2, 2), strides=(2, 2), name='pool5')(x)

    # Create model
    model = Model(image, x)

    # Load weights
    weights_path = get_file(WEIGHTS_FILE.format(depth), WEIGHTS_URL.format(depth), cache_subdir='models')
    model.load_weights(weights_path)

    return model

def preprocess(x):
    # Subtract VGG pixel means
    x[:, 0, :, :] -= PIXEL_MEANS['R']
    x[:, 1, :, :] -= PIXEL_MEANS['G']
    x[:, 2, :, :] -= PIXEL_MEANS['B']

    # Change channel order to VGG order (BGR)
    x = x[:, ::-1, :, :] # RGB to BGR

    return x

def deprocess(x):
    # Revert channel order to RGB
    x = x[:, ::-1, :, :] # BGR to RGB
    
    # Add VGG pixel means
    x[:, 0, :, :] += PIXEL_MEANS['R']
    x[:, 1, :, :] += PIXEL_MEANS['G']
    x[:, 2, :, :] += PIXEL_MEANS['B']

    return x