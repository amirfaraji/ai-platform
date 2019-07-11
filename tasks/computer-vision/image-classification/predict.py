from keras.models import load_model, model_from_json
from PIL import Image
from sklearn.model_selection import train_test_split

import cv2
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import neuralnetwork as nn
import numpy as np
import os
import sys

RANDOM_STATE = 100

def mri_tumor_predict():
    """
    Predict tumour on test set
    """
    model_path = sys.argv[1]
    weight_path = sys.argv[2]
    """
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    """
    model = load_model(model_path)
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
    
    pass

if __name__== "__main__":
   mri_tumor_predict()