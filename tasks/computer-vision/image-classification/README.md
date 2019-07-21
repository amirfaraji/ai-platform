# Image Classification  

Developed a Residual Attention Network for MRI brain tumor classification. 

## Task  

Small Kaggle dataset with binary classification of Brain Tumors. Dataset contains two folders of Brain MRI images and a total of 253 images. The folders are divided tumorous images and non-tumorous images and the objective of the neural network is to classify whether a Brain MRI image has a tumor or not. Dataset can be found [here].

[here]: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection  

## Model

Based on https://arxiv.org/pdf/1704.06904.pdf  

"*Residual Attention Network for Image Classification*" by Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang

The model used for this task was based on residual attention networks. Attention networks are developed as architectures which rapidly acquires the contexts of the image and selectively focus on the most important features. It can act as a regularization method on the output filters of a convolutional neural network by attenuating some feature maps and enhancing others. The residual attention networks implements two branches for their attention mechanism: a main branch and a mask branch. The main branch is generally the same architecture as any other stacked convolutional network, in this case the main branch is based on the ResNet-18 architecture. The mask branch of the architecture contains an autoencoder; the implementation is similar to the U-Net Convolutional Network. The input volume is the same for each the main branch and the mask branch. The masks branch generates a volume mask with values between 0 and 1. The output volumes of the mask branch are multplied with the output volume of the main branch. The first two convolutional blocks have attention modules and the last three convolutional blocks do not have any attention modules.



## File Information  

The main entry-point will download the model to the current working directory and the pretrained weights to the "pretrain_weights" folder. **Note:** The main entry-point must occur prior to the predict entry-point. The predict entry-point does not download any model or weights, and is therefore assumed that users have either trained their own network or downloaded the [model] and [weights]. The predict entry-point requires the model path and weight paths may be specified for the program.  
The pretrained weights have a validation accuracy of ~90%. Users can continue training from these weights or any other weights by setting the weight path (string) and flag (bool). If the dataset does not already exist, it will be downloaded upon executing either train or predict. The predict entry-point can predict on a single image or the testset of the data.

- neuralnetwork: The class definition of the neural network, builds or loads a network to the users specifications. Initialized with the hyperparameters. 
- train: Train data with training set (data split into three sets)
- predict: Predict the classes on the data's testset or on a single image
- model: Creates the model
- datagenerator: For artificial data generation, specifies the types of transforms used on the data and returns generator 
- hyperparameters: Hyperparameters specified upon execution of "mlflow run"
- metric: Contains Jaccard Index (metric not used)

[model]: https://drive.google.com/open?id=1USZ--9XrfO5oXwKJsh91lJnDtKnVV4n3
[weights]: https://drive.google.com/open?id=10CuubJJ7mhg3JsK0mp7sBvGCxNPH720X 

## Training
*Corresponding file - train.py*  
Splits the dataset into training, validation and test. A constant seed was used to split the data and the test set is used in the predict file. Trains the neural network with the training data (No K-folds cross validation).

### Parameters
- batch_size: Mini-Batch Size (default 8) 
- learning_rate: Learning Rate (default 0.001)
- epochs: Number of epochs to train (default 70) 
- optimizer: Training optimizer. Three possible values: *SGD*, *Adam*, *Nadam* (default 'SGD')
- patience: For *Early Stopping* callback. Specifies the number of epochs to continue after peak validation accuracy. If no there is no improvement in that time, *Early Stopping* is called (default 50)
- loss_function: Cost function for minimizing error of the specific data (default 'categorical_crossentropy')
- img_wid: Image Width (default 128) 
- img_hgt: Image Height (default 128) 
- img_channel: Image Channel (default 1) 
- num_of_classes: Number of classes in task (MRI Brain Task only has two classes - default 2)
- model_to_train: String specifying which model to train(Only 1 model currently - default 'ResNet-Attention')
- weight_path: Path to pre-trained weights (default 'weights.h5')
- load_weights_flag: Flag to load pretrained weights (default False)

## Predicting
*Corresponding file - predict.py*  
If a test image path is given for the image_path parameter the code will execute the function, *mri_tumor_predict_on_image*, and predicts the class of a single image. It will print the class to the console. If no test image is specified for the image_path parameter the code will Predicts Class on Test Set 

### Parameters
- model_path: Path to saved model (default 'ResNet-Attention-weights-improvement.h5')
- weight_path: Path to best trained weights (default 'ResNet-Attention-weights-improvement.h5')
- image_path : Path to a single image (no default)

### Example Command to Run
Command runs main entry point (training the neural network)
```
mlflow run .
```

Command runs predict entry point (predict class on test set)
```
mlflow run . -e predict
```

Command runs predict on a single image (predict class on test set)
```
mlflow run . -e predict -P image_path='images/Y1.jpg'
```
## Hardware
Keras API was employed with TensorFlow backend to implement the architecture. Training was done on Google Colab. Machine specifications are as followed:

- **GPU**: 1xTesla K80 , having 2496 CUDA cores, compute 3.7,  12GB(11.439GB Usable) GDDR5  VRAM
- **CPU**: 1xsingle core hyper threaded i.e(1 core, 2 threads) Xeon Processors @2.3Ghz (No Turbo Boost) , 45MB Cache
- **RAM**: ~12.6 GB Available
- **Disk**: ~320 GB Available 

