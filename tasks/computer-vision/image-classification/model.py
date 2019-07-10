from keras.applications.resnet50 import ResNet50
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Dropout  
from keras.layers import Flatten, GlobalAveragePooling2D, Input, MaxPooling2D, Multiply, UpSampling2D
from keras.models import Sequential, Model

import numpy as np
import tensorflow as tf


def common_layers(inputs, nb_outputs, kernel, strides=(1,1), padding='same', activation='relu'):
    """
    Order of Common Layers
    :param: input: Previous layer input
    :param nb_outputs: Output filter
    :param kernel: kernel size
    :param strides: stride length
    :param padding: padding
    :param activation: Activation Function  
    :return x: layer output
    """

    x = Conv2D(nb_outputs, kernel_size=kernel, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def residual_connection(input1, input2, nb_outputs, kernel, strides=(1,1), padding='same', activation='relu'):
    """
    Residual Connection module
    :param input1: Shortcut layer
    :param: input2: Previous layer input
    :param nb_outputs: Output filter
    :param kernel: kernel size
    :param strides: stride length
    :param padding: padding
    :param activation: Activation Function  
    :return res: layer output
    """

    if strides == (2,2):
        shortcut = Conv2D(nb_outputs, kernel_size=kernel, strides=strides, padding='same')(input1)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = Conv2D(nb_outputs, kernel_size=kernel, strides=(1,1), padding='same')(input1)
        shortcut = BatchNormalization()(shortcut)
    
    conv_layer = Conv2D(nb_outputs, kernel_size=kernel, strides=(1,1), padding='same')(input2)
    conv_layer = BatchNormalization()(conv_layer)

    res = Add()([conv_layer, shortcut])
    res = Activation(activation)(res)
    return res

def attention_module(inputs, nb_filts, out_filts):
    """
    Attention Mechanism with U-Net Architecture
    :param img_shape: Image shape
    :param nb_classes: For dense Layer output  
    :return model: Keras Model object
    """

    conv1 = common_layers(inputs, nb_filts, 3)
    conv1 = common_layers(conv1, nb_filts, 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = common_layers(pool1, 2*nb_filts, 3)
    conv2 = common_layers(conv2, 2*nb_filts, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = common_layers(pool2, 4*nb_filts, 3)
    conv3 = common_layers(conv3, 4*nb_filts, 3)
    
    up4 = common_layers(
        UpSampling2D(size=(2, 2))(conv3),
        2*nb_filts, 
        3
    )
    merge4 = Concatenate()([conv2, up4])
    conv4 = common_layers(merge4, 2*nb_filts, 3)
    conv4 = common_layers(conv4, 2*nb_filts, 3)

    up5 = common_layers(
        UpSampling2D(size=(2, 2))(conv4),
        nb_filts, 
        3
    )
    merge5 = Concatenate()([conv1, up5])
    conv5 = common_layers(merge5, nb_filts, 3)
    conv5 = common_layers(conv5, nb_filts, 3)

    outputs = common_layers(conv5, out_filts, 1, activation='sigmoid')

    return outputs

def build_res_attention_net(img_shape, nb_classes):
    """
    Build Residual Attention Network with the main branch constructed as ResNet18 
    :param img_shape: Image shape
    :param nb_classes: For dense Layer output  
    :return model: Keras Model object
    """

    inputs = Input(img_shape)

    # Block 1
    blk1 = common_layers(inputs, 64, kernel=(7, 7), strides=(2, 2), padding='same')

    # Block 2
    blk2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(blk1)
    blk2_1 = common_layers(blk2, 64, kernel=(1, 1), padding='same')
    shortcut2 = residual_connection(blk2, blk2_1, nb_outputs=256, kernel=(1,1))

    blk2_1 = common_layers(shortcut2, 64, kernel=(1, 1), padding='same')
    shortcut2 = residual_connection(shortcut2, blk2_1, nb_outputs=64, kernel=(1,1))

    attent2 = attention_module(blk2, 64, 64)
    multiply2 = Multiply()([attent2, shortcut2])
    h2 = Add()([multiply2, shortcut2])


    # Block 3
    blk3_1 = common_layers(h2, 128, kernel=(1, 1), strides=(2,2), padding='same')
    shortcut3 = residual_connection(h2, blk3_1, nb_outputs=128, kernel=(1,1), strides=(2,2))

    blk3_1 = common_layers(shortcut3, 128, kernel=(1, 1), padding='same')
    shortcut3 = residual_connection(shortcut3, blk3_1, nb_outputs=128, kernel=(1,1))
    
    attent3 = attention_module(MaxPooling2D(padding='same')(h2), 128, 128)
    multiply3 = Multiply()([attent3, shortcut3])
    h3 = Add()([multiply3, shortcut3])
    

    # Block 4
    blk4_1 = common_layers(h3, 256, kernel=(1, 1), strides=(2,2), padding='same')
    shortcut4 = residual_connection(h3, blk4_1, nb_outputs=256, kernel=(1,1), strides=(2,2))

    blk4_1 = common_layers(shortcut4, 256, kernel=(1, 1), padding='same')
    shortcut4 = residual_connection(shortcut4, blk4_1, nb_outputs=128, kernel=(1,1))
    

    # Block 5
    blk5_1 = common_layers(shortcut4, 512, kernel=(1, 1), strides=(2,2), padding='same')
    shortcut5 = residual_connection(shortcut4, blk5_1, nb_outputs=128, kernel=(1,1), strides=(2,2))

    blk3_1 = common_layers(shortcut5, 512, kernel=(1, 1), padding='same')
    shortcut5 = residual_connection(shortcut5, blk5_1, nb_outputs=128, kernel=(1,1))

    # FCN
    x = GlobalAveragePooling2D()(shortcut5)
    predictions = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model
