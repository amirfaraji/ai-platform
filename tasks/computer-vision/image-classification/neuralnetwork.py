from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from os import path

import model as m


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
        self.classes = hyperparameters["classes"]
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

        weight_checkpoint_path= self.model_to_train+"-weights-improvement.h5"
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

    def train(self, x_train, y_train, x_val, y_val):
        """
        Train without generator
        :param x_train: Training img
        :param x_val:  Validation img
        :param y_train: Training label
        :param y_val: Validation label
        :return: 
        """

        self._build_model()

        self.model.summary()

        if (self.load_weights_flag): 
            if path.exists(self.weight_path):
                self.model.load_weights(self.weight_path)
            else: 
                raise ValueError("Weight path does not exist")
        
        self.model.compile(loss=self.loss, optimizer=self.opti, metrics=["accuracy", self.jaccard_index])


        self.history = self.model.fit(x_train, 
            y_train, 
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(x_val, y_val),
            callbacks=[self.checkpoint, self.earlystopping]
        )

        pass

    def train_generator(self, train_gen, valid_gen):
        """
        Train with generator
        :param train_gen: Generator for the
        :param valid_gen:  
        :return: 
        """

        self._build_model()

        self.model.summary()

        if (self.load_weights_flag): 
            if path.exists(self.weight_path):
                self.model.load_weights(self.weight_path)
            else: 
                raise ValueError("Weight path does not exist")
        
        self.model.compile(loss=self.loss, optimizer=self.opti, metrics=["accuracy", self.jaccard_index])

        self.history = self.model.fit_generator(
            train_gen,
            steps_per_epoch=len(train_gen), 
            epochs=self.epochs, 
            validation_data=valid_gen,
            validation_steps=len(valid_gen),
            callbacks=[self.checkpoint, self.earlystopping])
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
        self.model.compile(loss=self.loss, optimizer=self.opti, metrics=["accuracy", self.jaccard_index])
        
        scores = self.model.evaluate(x_test, y_test, verbose=1)
        return scores


    def jaccard_index(self, y_true, y_pred, smooth=1e-12):
        """
        Metric - Jaccard Index
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param smooth:   
        :return: Jaccard index 
        """

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)