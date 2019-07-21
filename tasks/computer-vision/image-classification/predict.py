from google_drive_downloader import GoogleDriveDownloader as gdd
from keras.models import load_model, model_from_json
from PIL import Image
from sklearn.model_selection import train_test_split

import cv2
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import neuralnetwork as nn
import numpy as np
import os
import sys

RANDOM_STATE = 100

def mri_tumor_predict_on_set(model_path, weight_path):
    """
    Predict tumor on test set
    :param model_path: Path to a saved model
    :param weight_path: Path to pretrained weights 
    :param smooth:   
    :return:
    """
    
    if os.path.exists(model_path):
        model = load_model(model_path)
    else: 
        raise ValueError("Model Path does not exist")

    if os.path.exists(weight_path):
        model.load_weights(weight_path)
    

    if K.image_data_format() == 'channels_first':
        _, img_channels, img_wid, img_hgt = model.layers[0].get_input_shape_at(0)
    else:
        _, img_wid, img_hgt, img_channels = model.layers[0].get_input_shape_at(0)

    _, num_of_classes = model.layers[-1].output_shape

    img_dir = ["./brain-mri-images-for-brain-tumor-detection/no/", "./brain-mri-images-for-brain-tumor-detection/yes/"]
    file_count = 0
    _, _, no_files = next(os.walk(img_dir[0]))
    file_count += len(no_files)
    _, _, yes_files = next(os.walk(img_dir[1]))
    file_count += len(yes_files)

    img_data = np.zeros((file_count, img_wid, img_hgt))
    label_data = np.zeros((file_count, num_of_classes))
    i = 0
    for file in no_files:
        im = np.asarray(Image.open(img_dir[0]+file).convert('L'))
        img_data[i,:,:] = cv2.resize(im, dsize=(img_wid, img_hgt), interpolation = cv2.INTER_CUBIC)
        label_data[i,0] = 1
        i += 1
    
    for file in yes_files:
        im = np.asarray(Image.open(img_dir[1]+file).convert('L'))
        img_data[i,:,:] = cv2.resize(im, dsize=(img_wid, img_hgt), interpolation = cv2.INTER_CUBIC)
        label_data[i,1] = 1
        i += 1
    
    if K.image_data_format() == 'channels_first':
        img_data = np.expand_dims(img_data,axis=1)
    else:
        img_data = np.expand_dims(img_data,axis=3)


    x_train_val, x_test, y_train_val, y_test = train_test_split(img_data, label_data, test_size=0.15, random_state=RANDOM_STATE)

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.15, random_state=RANDOM_STATE)
    
    scores = model.evaluate(x_test, y_test, verbose=1)

    print("Test Loss: ", scores[0])
    print("Test Accuracy: ", scores[1])
    with mlflow.start_run():
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("weight_path", weight_path)
        mlflow.log_param("function", "predict_on_set")
        # log metrics
        mlflow.log_metric("test_loss", scores[0])
        mlflow.log_metric("test_accuracy", scores[1])

        mlflow.keras.log_model(model, "Trained-Model")
    pass

def mri_tumor_predict_on_image(model_path, weight_path, image_path):
    """
    Predict tumor on a single image
    :param model_path: Path to a saved model
    :param weight_path: Path to pretrained weights 
    :param test_image: Image to predict class   
    :return:
    """
    if os.path.exists(model_path):
        model = load_model(model_path)
    else: 
        raise ValueError("Model Path does not exist")

    if os.path.exists(weight_path):
        model.load_weights(weight_path)


    if K.image_data_format() == 'channels_first':
        _, img_channels, img_wid, img_hgt = model.layers[0].get_input_shape_at(0)
    else:
        _, img_wid, img_hgt, img_channels = model.layers[0].get_input_shape_at(0)


    if os.path.exists(image_path):
        im = np.asarray(Image.open(image_path).convert('L'))
        im = cv2.resize(im, dsize=(img_wid, img_hgt), interpolation = cv2.INTER_CUBIC)
        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=3)
    else: 
        raise ValueError("Weights Path does not exist")

    class_names = ["No tumor", "Tumor"]
    result = model.predict(im)
    print(class_names[result.argmax(axis=-1)[0]], "is present")

    with mlflow.start_run():
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("weight_path", weight_path)
        mlflow.log_param("image_path", image_path)
        mlflow.log_param("function", "predict_on_image")
        # log metrics
        mlflow.log_metric("Class", result.argmax(axis=-1)[0])

        mlflow.keras.log_model(model, "Trained-Model")
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
    
    model_path = sys.argv[1]
    weight_path = sys.argv[2]
    image_path = sys.argv[3]

    if os.path.exists(image_path):
        mri_tumor_predict_on_image(model_path, weight_path, image_path)
    else:
        print("Image path was not found. Predicting on test set")
        mri_tumor_predict_on_set(model_path, weight_path)