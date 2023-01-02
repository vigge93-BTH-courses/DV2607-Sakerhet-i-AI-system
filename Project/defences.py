from art.defences.preprocessor import SpatialSmoothingTensorFlowV2, GaussianAugmentation
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def getSmoothedImage(images):
    ss_filter = SpatialSmoothingTensorFlowV2(window_size=3)
    ss_image, _ = ss_filter(images)
    return ss_image


def ImageAugmentation(x_train, y_train, x_test, y_test):
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    # datagen.fit(x_train)

    def fit_model(model):
        model.fit(datagen.flow(x_train, y_train, batch_size=128), steps_per_epoch=len(x_train) / 128, epochs=70, validation_data=(x_test, y_test))
        return model
    return fit_model


def GaussianNoise(images, lables, fit=True):
    ga = GaussianAugmentation(sigma=0.05, augmentation=fit, clip_values=(0, 1), apply_fit=fit, apply_predict=(not fit), ratio=0.5)
    return ga(images, lables)

if __name__ == '__main__':
    from data import getCifar10
    import matplotlib.pyplot as plt
    x_train, y_train, x_test, y_test = getCifar10()
    img = x_train[1]
    plt.imshow(img)
    plt.show()
    noise, labels = GaussianNoise(np.array([img]), None, False)
    plt.imshow(noise[0])
    plt.show()
    smooth = getSmoothedImage(img)
    plt.imshow(smooth[0])
    plt.show()
