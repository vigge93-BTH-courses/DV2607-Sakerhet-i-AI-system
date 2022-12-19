from keras import models
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Softmax, BatchNormalization
from data import getCifar10, saveModel
import tensorflow as tf


def getModelVGG16():
    L2 = 0.0005
    inputs = Input(shape=(32, 32, 3))
    conv1 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2), input_shape=(32, 32, 3))(inputs)
    norm1 = BatchNormalization()(conv1)
    conv2 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm1)
    norm2 = BatchNormalization()(conv2)
    maxpool1 = MaxPooling2D(pool_size=2, strides=2, padding='same')(norm2)
    conv3 = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(maxpool1)
    norm3 = BatchNormalization()(conv3)
    conv4 = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm3)
    norm4 = BatchNormalization()(conv4)
    maxpool2 = MaxPooling2D(pool_size=2, strides=2, padding='same')(norm4)
    conv5 = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(maxpool2)
    norm5 = BatchNormalization()(conv5)
    conv6 = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm5)
    norm6 = BatchNormalization()(conv6)
    conv7 = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm6)
    norm7 = BatchNormalization()(conv7)
    maxpool3 = MaxPooling2D(pool_size=2, strides=2, padding='same')(norm7)
    conv8 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(maxpool3)
    norm8 = BatchNormalization()(conv8)
    conv9 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm8)
    norm9 = BatchNormalization()(conv9)
    conv10 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm9)
    norm10 = BatchNormalization()(conv10)
    maxpool4 = MaxPooling2D(pool_size=2, strides=2, padding='same')(norm10)
    conv11 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(maxpool4)
    norm11 = BatchNormalization()(conv11)
    conv12 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm11)
    norm12 = BatchNormalization()(conv12)
    conv13 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm12)
    norm13 = BatchNormalization()(conv13)
    maxpool5 = MaxPooling2D(pool_size=2, strides=2, padding='same')(norm13)
    flatten = Flatten()(maxpool5)
    dense1 = Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(flatten)
    norm14 = BatchNormalization()(dense1)
    dense2 = Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm14)
    norm15 = BatchNormalization()(dense2)
    dense3 = Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(L2))(norm15)

    model = models.Model(
        inputs=inputs,
        outputs=dense3
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
    return model


def fit(model: models.Model, train_data, train_labels, test_data, test_labels):
    history = model.fit(train_data, train_labels, epochs=250, batch_size=128, validation_data=(test_data, test_labels))
    return history


if __name__ == '__main__':
    train_data, train_lables, test_data, test_labels = getCifar10()
    # train_data, train_lables, test_data, test_labels = preprocess_data(data['train_data'], data['test_data'])
    vgg16_model = getModelVGG16()
    history = fit(vgg16_model, train_data, train_lables, test_data, test_labels)
    saveModel(vgg16_model, 'Project/models/vgg16')
    print(history.history)
