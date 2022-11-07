from keras import models
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, AveragePooling2D, Flatten, Softmax, GlobalAveragePooling2D
from data import getCifar10, preprocess_data
import tensorflow as tf


def getModel():
    inputs = Input(shape=(32, 32, 3))
    conv1 = Conv2D(192, padding='same', kernel_size=5, input_shape=(32, 32, 3))(inputs)
    conv2 = Conv2D(160, padding='same', kernel_size=1)(conv1)
    conv3 = Conv2D(96, padding='same', kernel_size=1)(conv2)
    maxpool1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv3)
    dropout1 = Dropout(0.5)(maxpool1)
    conv4 = Conv2D(192, padding='same', kernel_size=5)(dropout1)
    conv5 = Conv2D(192, padding='same', kernel_size=5)(conv4)
    conv6 = Conv2D(192, padding='same', kernel_size=5)(conv5)
    avgpool1 = AveragePooling2D(pool_size=3, strides=2, padding='same')(conv6)
    dropout2 = Dropout(0.5)(avgpool1)
    conv7 = Conv2D(192, padding='same', kernel_size=3)(dropout2)
    conv8 = Conv2D(192, padding='same', kernel_size=1)(conv7)
    conv9 = Conv2D(10, padding='same', kernel_size=1)(conv8)
    avgpool2 = GlobalAveragePooling2D()(conv9)
    flatten = Flatten()(avgpool2)
    softmax = Softmax()(flatten)

    model = models.Model(
        inputs=inputs,
        outputs=softmax
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
    return model


def fit(model: models.Model, train_data, train_labels, test_data, test_labels):
    history = model.fit(train_data, train_labels, epochs=200, batch_size=128, validation_data=(test_data, test_labels))
    return history


if __name__ == '__main__':
    data = getCifar10()
    train_data, train_lables, test_data, test_labels = preprocess_data(data['train_data'], data['test_data'])
    nin_model = getModel()
    history = fit(nin_model, train_data, train_lables, test_data, test_labels)
    print(history.history)
