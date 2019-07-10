from keras.preprocessing.image import ImageDataGenerator

SEED = 100
TRAIN_AUG = dict(
    shear_range=0.05,
    zoom_range=0.10, 
    rotation_range=90
)


def train_generator(x_train, y_train, batch_size):
    """
    Data generator for Training set
    :param x_train: Training set of image data
    :param y_train: Training set of labels
    :param batch_size: Size of mini-batch
    :param aug_dict: dictionary of parameter ranges for augmentation
    :return train_data: Train generator
    """

    train_img_datagen = ImageDataGenerator(**TRAIN_AUG)
    train_data = train_img_datagen.flow(
        x_train, 
        y=y_train, 
        batch_size=batch_size, 
        shuffle=True, 
        seed=SEED
    )
    return train_data

def val_generator(x_valid, y_valid, batch_size):
    """
    Data generator for Validation set
    :param x_valid: Validation set of image data
    :param y_valid: Validation set of labels
    :param batch_size: Size of mini-batch
    :return val_data: Val generator
    """

    validation_datagen = ImageDataGenerator()
    val_data = validation_datagen.flow(
        x_valid, 
        y_valid, 
        batch_size=batch_size,
        shuffle=True, 
        seed=SEED
    )
    return val_data