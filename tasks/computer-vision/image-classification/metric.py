from keras import backend as K

def jaccard_index(y_true, y_pred, smooth=1e-12):
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