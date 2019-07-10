from hyperparameters import mnist_hyperparameters, mri_hyperparameters
from keras.datasets import mnist
from PIL import Image
from sklearn.model_selection import train_test_split

import cv2
import datagenerator as dg
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import neuralnetwork as nn
import numpy as np
import os

RANDOM_STATE = 100


def mri_tumor_detection():
    """
    Small dataset with binary classification For Training and Testing purposes
    https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
    Validation Set Accuracy: ~88%
    Test Set Accuracy ~84% 
    """
    img_dir = ["./brain-mri-images-for-brain-tumor-detection/no/", "./brain-mri-images-for-brain-tumor-detection/yes/"]
    file_count = 0
    _, _, no_files = next(os.walk(img_dir[0]))
    file_count += len(no_files)
    _, _, yes_files = next(os.walk(img_dir[1]))
    file_count += len(yes_files)

    img_data = np.zeros((file_count, mri_hyperparameters["img_wid"], mri_hyperparameters["img_hgt"]))
    label_data = np.zeros((file_count, mri_hyperparameters["num_of_classes"]))
    i = 0
    for file in no_files:
        im = np.asarray(Image.open(img_dir[0]+file).convert('L'))
        img_data[i,:,:] = cv2.resize(im, dsize=(mri_hyperparameters["img_wid"], mri_hyperparameters["img_hgt"]), interpolation = cv2.INTER_CUBIC)
        label_data[i,0] = 1
        i += 1
    
    for file in yes_files:
        im = np.asarray(Image.open(img_dir[1]+file).convert('L'))
        img_data[i,:,:] = cv2.resize(im, dsize=(mri_hyperparameters["img_wid"], mri_hyperparameters["img_hgt"]), interpolation = cv2.INTER_CUBIC)
        label_data[i,1] = 1
        i += 1
    
    if K.image_data_format() == 'channels_first':
        img_data = np.expand_dims(img_data,axis=1)
    else:
        img_data = np.expand_dims(img_data,axis=3)


    x_train_val, x_test, y_train_val, y_test = train_test_split(img_data, label_data, test_size=0.15, random_state=RANDOM_STATE)

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.15, random_state=RANDOM_STATE)
    
    train_data = dg.train_generator(x_train, y_train, mri_hyperparameters["batch_size"])
    val_data = dg.val_generator(x_val, y_val, mri_hyperparameters["batch_size"])

    network = nn.NeuralNetwork(**mri_hyperparameters)
    #network.train(x_train, y_train, x_val, y_val)
    network.train_generator(train_data, val_data)
    scores = network.predict(x_test, y_test)
    print(scores)
    pass

def mnist_test():
    """
    Quick MNIST test to test network - No test set only training and validation set 
    Validation Set Accuracy ~98+%
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        train_shape = (x_train.shape[0], mnist_hyperparameters["img_channel"], mnist_hyperparameters["img_wid"], mnist_hyperparameters["img_hgt"])
        test_shape = (x_test.shape[0], mnist_hyperparameters["img_channel"], mnist_hyperparameters["img_wid"], mnist_hyperparameters["img_hgt"])
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
        resized_x_train = np.zeros((train_shape[0],train_shape[2:]))
        resized_x_test = np.zeros((train_shape[0],train_shape[2:]))    
        for i in range(len(resized_x_train)):
            resized_x_train[i,:,:] = cv2.resize(x_train[i,0,:,:], dsize=(train_shape[2:]), interpolation = cv2.INTER_CUBIC)
        for i in range(len(resized_x_test)):
            resized_x_test[i,:,:] = cv2.resize(x_test[i,0,:,:], dsize=(test_shape[2:]), interpolation = cv2.INTER_CUBIC)
        resized_x_train = np.expand_dims(resized_x_train,axis=1)
        resized_x_test = np.expand_dims(resized_x_test,axis=1)
    else:
        train_shape = (x_train.shape[0], mnist_hyperparameters["img_wid"], mnist_hyperparameters["img_hgt"], mnist_hyperparameters["img_channel"])
        test_shape = (x_test.shape[0], mnist_hyperparameters["img_wid"], mnist_hyperparameters["img_hgt"], mnist_hyperparameters["img_channel"])
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        resized_x_train = np.zeros((train_shape[0:3]))
        resized_x_test = np.zeros((test_shape[0:3]))    
        for i in range(len(resized_x_train)):
            resized_x_train[i,:,:] = cv2.resize(x_train[i,:,:,0], dsize=(train_shape[1:3]), interpolation = cv2.INTER_CUBIC)
        for i in range(len(resized_x_test)):
            resized_x_test[i,:,:] = cv2.resize(x_test[i,:,:,0], dsize=(test_shape[1:3]), interpolation = cv2.INTER_CUBIC)
        resized_x_train = np.expand_dims(resized_x_train,axis=3)
        resized_x_test = np.expand_dims(resized_x_test,axis=3)


    y_train = keras.utils.to_categorical(y_train, mnist_hyperparameters["num_of_classes"])
    y_test = keras.utils.to_categorical(y_test, mnist_hyperparameters["num_of_classes"])
    resized_x_train = x_train.astype('float32')
    resized_x_test = x_test.astype('float32')
    resized_x_train /= 255
    resized_x_test /= 255



    network = nn.NeuralNetwork(**mnist_hyperparameters)
    network.train(resized_x_train, y_train, resized_x_test, y_test)

    pass


if __name__== "__main__":
   #mnist_test()
   mri_tumor_detection()