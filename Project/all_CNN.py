import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib as plt
from keras import layers, models
import data as d


def get_cnn(summary=False):
    model = models.Sequential()
    model.add(layers.Conv2D(96, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(96, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(96, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2D(192, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(192, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(192, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2D(192, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2D(192, kernel_size=1, strides=1, padding='valid', activation='relu'))
    model.add(layers.Conv2D(10, kernel_size=1, strides=1, padding='valid', activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Activation('softmax'))

    if summary is True:
        model.summary()
    return model


def compile_fit(model, x, y, x_test, y_test, epochs=100, batch_size=128):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=10)])
    history = model.fit(x, y, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)
    return history


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = d.getCifar10()
    model = get_cnn(summary=True)
    history = compile_fit(model, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=128)
    # plt.plot(history.history['accuracy'], label='accuracy')