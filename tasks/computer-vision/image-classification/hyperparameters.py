mnist_hyperparameters = {
    "batch_size"        : 128,
    "learning_rate"     : 0.001,
    "epochs"            : 10,
    "optimizer"         : "SGD",
    "patience"          : 4,
    "loss_function"     : "categorical_crossentropy",
    "img_wid"           : 32,
    "img_hgt"           : 32,
    "img_channel"       : 1,
    "num_of_classes"    : 10,
    "classes"           : ['',''],
    "model_to_train"    : 'ResNet-Attention', 
    "weight_path"       : './weights_folder/some_weights.h5',
    "load_weights_flag"  : False
}

mri_hyperparameters = {
    "batch_size"        : 8,
    "learning_rate"     : 0.001,
    "epochs"            : 70,
    "optimizer"         : "SGD",
    "patience"          : 50,
    "loss_function"     : "categorical_crossentropy",
    "img_wid"           : 128,
    "img_hgt"           : 128,
    "img_channel"       : 1,
    "num_of_classes"    : 2,
    "classes"           : ['',''],
    "model_to_train"    : 'ResNet-Attention', 
    "weight_path"       : './weights_folder/some_weights.h5',
    "load_weights_flag"  : False
}
