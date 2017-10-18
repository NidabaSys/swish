from keras import backend as K

def swish(x):
    return K.sigmoid(x) * x
