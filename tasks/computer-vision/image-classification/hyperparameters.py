import sys
if len(sys.argv) == 14:
    mri_hyperparameters = {
        "batch_size"        : int(sys.argv[1]),
        "learning_rate"     : float(sys.argv[2]),
        "epochs"            : int(sys.argv[3]),
        "optimizer"         : sys.argv[4],
        "patience"          : int(sys.argv[5]),
        "loss_function"     : sys.argv[6],
        "img_wid"           : int(sys.argv[7]),
        "img_hgt"           : int(sys.argv[8]),
        "img_channel"       : int(sys.argv[9]),
        "num_of_classes"    : int(sys.argv[10]),
        "model_to_train"    : sys.argv[11], 
        "weight_path"       : sys.argv[12],
        "load_weights_flag" : sys.argv[13]
    }
else:
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