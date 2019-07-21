from google_drive_downloader import GoogleDriveDownloader as gdd
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from os import path

import mlflow
import mlflow.keras
import model as m
import numpy as np


class NeuralNetwork:
    """
    Neural Network Class for Image Classification.
    """

    def __init__(self, **hyperparameters):
        """
        Neural Network Initialization with Hyperparameters
        :param batch_size: Size of mini-batch
        :param epochs: Number of epochs for training
        :param learning_rate: 
        :param classes: Class labels
        :param model_to_train: String for which model to train
        :param weights_path: Path to trained weights 
        :param load_weights_flag: Flag to load weights 
        :return: 
        """
        self.batch_size = hyperparameters["batch_size"]
        self.epochs = hyperparameters["epochs"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.patience = hyperparameters["patience"]
        self.optimizer = hyperparameters["optimizer"]
        self.loss = hyperparameters["loss_function"]
        self.num_of_classes = hyperparameters["num_of_classes"]
        self.model_to_train = hyperparameters["model_to_train"]
        self.weight_path = hyperparameters["weight_path"]
        
        self.load_weights_flag = hyperparameters["load_weights_flag"]

        if K.image_data_format == 'channel_first':
            self.img_shape = (hyperparameters["img_channel"], hyperparameters["img_wid"], hyperparameters["img_hgt"])
        else:
            self.img_shape = (hyperparameters["img_wid"], hyperparameters["img_hgt"], hyperparameters["img_channel"])
        
        if self.optimizer == "Adam":
            self.opti = optimizers.adam(lr=self.learning_rate)
        elif self.optimizer == "Nadam":
            self.opti = optimizers.nadam(lr=self.learning_rate)
        else: 
            self.opti = optimizers.sgd(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

        weight_checkpoint_path= "./saved_weights/"+self.model_to_train+"-weights-improvement.h5"
        self.checkpoint = ModelCheckpoint(
            weight_checkpoint_path, 
            monitor='val_acc', 
            verbose=1, 
            save_best_only=True,
            mode='max'
        )

        self.earlystopping = EarlyStopping(
            monitor='val_acc',
            patience=self.patience
        )
        pass

    def _build_model(self,):
        """
        Build Network from model.py 
        :param :  
        :return: 
        """

        if self.model_to_train == "ResNet-Attention":
            self.model = m.build_res_attention_net(self.img_shape, self.num_of_classes)
        else:
            raise ValueError("No match found for " + self.model_to_train)

        pass

    def _load_model(self,):
        """
        Loads model  
        :param :  
        :return: 
        """
        model_path = './model.h5'
        print("Downloading Model as", model_path)
        gdd.download_file_from_google_drive(
            file_id='1USZ--9XrfO5oXwKJsh91lJnDtKnVV4n3',
            dest_path=model_path,
            unzip=False
        )
        print("Downloading pretrained weights")
        gdd.download_file_from_google_drive(
            file_id='10CuubJJ7mhg3JsK0mp7sBvGCxNPH720X',
            dest_path='./pretrain_weights/pretrained-RAN-weights.h5',
            unzip=False
        )
        self.model = load_model(model_path)

    def train(self, x_train, y_train, x_val, y_val):
        """
        Train without generator
        :param x_train: Training img
        :param x_val:  Validation img
        :param y_train: Training label
        :param y_val: Validation label
        :return: 
        """
        # Builds model from scratch
        # self._build_model()

        # Downloads and loads model on runtime
        self._load_model()

        self.model.summary()

        if self.load_weights_flag == "True": 
            if path.exists(self.weight_path):
                self.model.load_weights(self.weight_path)
            else: 
                raise ValueError("Weight path does not exist")
        
        self.model.compile(loss=self.loss, optimizer=self.opti, metrics=["accuracy"])


        self.history = self.model.fit(x_train, 
            y_train, 
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(x_val, y_val),
            callbacks=[self.checkpoint, self.earlystopping]
        )

        # Save the last run model
        # self.model.save(self.model_to_train+"-model.h5")

        with mlflow.start_run():
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("optimizer", self.optimizer)
            mlflow.log_param("loss_function", self.loss)
            mlflow.log_param("num_of_classes", self.num_of_classes)
            mlflow.log_param("model_to_train", self.model_to_train)
            mlflow.log_param("weight_path", self.weight_path)
            mlflow.log_param("load_weights_flag", self.load_weights_flag)
            # log metrics
            mlflow.log_metric("training_loss", self.history.history['loss'][-1])
            mlflow.log_metric("training_acc",  self.history.history['acc'][-1])
            mlflow.log_metric("validation_loss", self.history.history['val_loss'][-1])
            mlflow.log_metric("validation_acc",  self.history.history['val_acc'][-1])
            mlflow.log_metric("peak_validation_acc", np.amax(self.history.history['val_acc']))


            # log artifacts (matplotlib images for loss/accuracy)
            # mlflow.log_artifacts(image_dir)
            #log model
            mlflow.keras.log_model(self.model, self.model_to_train+"-model")

        pass

    def train_generator(self, train_gen, valid_gen):
        """
        Train with generator
        :param train_gen: Generator for the
        :param valid_gen:  
        :return: 
        """
        # Builds model from scratch
        # self._build_model()

        # Downloads and loads model on runtime
        self._load_model()

        self.model.summary()

        if self.load_weights_flag == "True": 
            if path.exists(self.weight_path):
                self.model.load_weights(self.weight_path)
            else: 
                raise ValueError("Weight path does not exist")
        
        self.model.compile(loss=self.loss, optimizer=self.opti, metrics=["accuracy"])

        self.history = self.model.fit_generator(
            train_gen,
            steps_per_epoch=len(train_gen), 
            epochs=self.epochs, 
            validation_data=valid_gen,
            validation_steps=len(valid_gen),
            callbacks=[self.checkpoint, self.earlystopping]
        )
        
        # Save the last run model
        # self.model.save(self.model_to_train+"-model.h5")

        with mlflow.start_run():
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("optimizer", self.optimizer)
            mlflow.log_param("loss_function", self.loss)
            mlflow.log_param("num_of_classes", self.num_of_classes)
            mlflow.log_param("model_to_train", self.model_to_train)
            mlflow.log_param("weight_path", self.weight_path)
            mlflow.log_param("load_weights_flag", self.load_weights_flag)
            # log metrics
            mlflow.log_metric("training_loss", self.history.history['loss'][-1])
            mlflow.log_metric("training_acc",  self.history.history['acc'][-1])
            mlflow.log_metric("validation_loss", self.history.history['val_loss'][-1])
            mlflow.log_metric("validation_acc",  self.history.history['val_acc'][-1])
            mlflow.log_metric("peak_validation_acc", np.amax(self.history.history['val_acc']))


            # log artifacts (matplotlib images for loss/accuracy)
            # mlflow.log_artifacts(image_dir)
            #log model
            mlflow.keras.log_model(self.model, self.model_to_train+"-model")
        pass

    def predict(self, x_test, y_test):
        """
        Prediction of the test set, calls the evaluate method
        :param x_test: Test images 
        :param y_test: Test labels 
        :return scores: Returns the scores of the test set 
        """

        if path.exists(self.model_to_train+'-weights-improvement.h5'):
            self.model.load_weights(self.model_to_train+'-weights-improvement.h5')
        self.model.compile(loss=self.loss, optimizer=self.opti, metrics=["accuracy"])
        
        scores = self.model.evaluate(x_test, y_test, verbose=1)
        return scores

    def save_model(self,):
        # serialize model to JSON
        model_json = self.model.to_json()

        with open(self.model_to_train + "-model.json", "w") as json_file:
            json_file.write(model_json)
        pass
