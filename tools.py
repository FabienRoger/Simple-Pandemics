import numpy as np
import keras.backend as K

def sigmoid(z):
    return 1/(1+np.exp(-z))

def abs_diff(y_true,y_pred):
    return K.mean(K.abs(y_pred - y_true))