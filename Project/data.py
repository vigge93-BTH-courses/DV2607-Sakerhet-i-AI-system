# ignore: E501
import pickle
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
from keras import models
import tensorflow as tf


def unpickle(file: str):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def getCifar10() -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Returns (x_train, y_train, x_test, y_test)"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Return both
    return x_train, y_train, x_test, y_test


def saveModel(model: models.Model, filepath: str):
    model.save(filepath=filepath)


def loadModel(filepath):
    return tf.keras.models.load_model(filepath)


if __name__ == '__main__':
    data = getCifar10()
    print(data[1][0])
