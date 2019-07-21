from google_drive_downloader import GoogleDriveDownloader as gdd
from hyperparameters import mri_hyperparameters
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
    Small Kaggle dataset with binary classification For Training and Testing purposes
    https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
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

    pass


if __name__== "__main__":

    if not os.path.exists('./brain-mri-images-for-brain-tumor-detection'):
        print("Downloading 'Brain MRI Dataset'")
        gdd.download_file_from_google_drive(
            file_id='1mpxHKunUg01DH6_xD8ozzianXPObHFJ2',
            dest_path='./brain-mri-images-for-brain-tumor-detection.zip',
            unzip=True
        )
    else: 
        print("Dataset already exists")

    mri_tumor_detection()