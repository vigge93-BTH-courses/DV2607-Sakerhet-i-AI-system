from keras import models
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, AveragePooling2D, Flatten, Softmax, GlobalAveragePooling2D, BatchNormalization
from data import getCifar10, saveModel
import tensorflow as tf
import matplotlib.pyplot as plt


def getModelNiN():
    L2 = 0.0001
    inputs = Input(shape=(32, 32, 3))
    conv1 = Conv2D(192, padding='same', kernel_size=5, input_shape=(32, 32, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(inputs)
    norm1 = BatchNormalization()(conv1)
    conv2 = Conv2D(160, padding='same', kernel_size=1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm1)
    norm2 = BatchNormalization()(conv2)
    conv3 = Conv2D(96, padding='same', kernel_size=1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm2)
    norm3 = BatchNormalization()(conv3)
    maxpool1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(norm3)
    dropout1 = Dropout(0.5)(maxpool1)
    conv4 = Conv2D(192, padding='same', kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(dropout1)
    norm4 = BatchNormalization()(conv4)
    conv5 = Conv2D(192, padding='same', kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm4)
    norm5 = BatchNormalization()(conv5)
    conv6 = Conv2D(192, padding='same', kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm5)
    norm6 = BatchNormalization()(conv6)
    avgpool1 = AveragePooling2D(pool_size=3, strides=2, padding='same')(norm6)
    dropout2 = Dropout(0.5)(avgpool1)
    conv7 = Conv2D(192, padding='same', kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(dropout2)
    norm7 = BatchNormalization()(conv7)
    conv8 = Conv2D(192, padding='same', kernel_size=1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm7)
    norm8 = BatchNormalization()(conv8)
    conv9 = Conv2D(10, padding='same', kernel_size=1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm8)
    norm9 = BatchNormalization()(conv9)
    avgpool2 = GlobalAveragePooling2D()(norm9)
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
    train_data, train_lables, test_data, test_labels = getCifar10()
    nin_model = getModelNiN()
    history = fit(nin_model, train_data, train_lables, test_data, test_labels)
    saveModel(nin_model, 'Project/models/nin')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
