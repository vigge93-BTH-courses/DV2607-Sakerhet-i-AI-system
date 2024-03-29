from DE import perturbImage
import NiN
import CNN
import VGG16
from data import getCifar10, saveModel, loadModel
import numpy as np
from defences import getSmoothedImage, ImageAugmentation, GaussianNoise
import matplotlib.pyplot as plt
from keras import models, backend
from os import makedirs
import time
import tensorflow as tf


def attack(models: "dict[str, models.Model]", test_data: "np.ndarray", test_labels: "np.ndarray"):
    with open('attacked.csv', 'r') as f:
        attackedList = [int(row.strip()) for row in f]
    mask = np.ones(len(test_labels), dtype=bool)
    mask[attackedList] = False
    test_data = test_data[mask]
    test_labels = test_labels[mask]
    with open('results.csv', 'a') as results_csv:
        print('ImageId,Label,ModelName,OriginalPrediction,AttackedPrediction,SmoothedPrediction,NoisedPrediction,Time', file=results_csv)
        for imageId in np.random.choice(np.arange(len(test_data)), size=1000, replace=False):
            image = test_data[imageId]
            label = np.argmax(test_labels[imageId])

            makedirs(f'data/original/', exist_ok=True)
            makedirs(f'images/original/', exist_ok=True)

            plt.imsave(f'images/original/{imageId}_label_{label}.png', image)
            np.save(f'data/original/{imageId}', image)

            for model in models:
                print('=' * 25)
                print(f'Model: {model}')
                print('=' * 25)
                makedirs(f'data/{model}/', exist_ok=True)
                makedirs(f'images/{model}/', exist_ok=True)

                originalPrediction = np.argmax(models[model].predict(np.array([image]), verbose=0))
                print(f'ImageId: {imageId}, Original: {originalPrediction},')

                start_time = time.perf_counter_ns()
                attacked = perturbImage(image, label, models[model])
                end_time = time.perf_counter_ns()
                attackedPrediction = np.argmax(models[model].predict(np.array([attacked]), verbose=0))

                plt.imsave(f'images/{model}/{imageId}_label_{attackedPrediction}.png', attacked)
                np.save(f'data/{model}/{imageId}', attacked)

                smoothed = getSmoothedImage(np.array([attacked]))
                smoothedPrediction = np.argmax(models[model].predict(smoothed, verbose=0))
                plt.imsave(f'images/{model}/{imageId}_label_{smoothedPrediction}_smoothed.png', smoothed[0])
                np.save(f'data/{model}/{imageId}_smoothed', smoothed[0])

                noised, _ = GaussianNoise(np.array([attacked]), None, False)
                noisedPrediction = np.argmax(models[model].predict(noised, verbose=0))
                plt.imsave(f'images/{model}/{imageId}_label_{noisedPrediction}_noised.png', noised[0])
                np.save(f'data/{model}/{imageId}_noised', noised[0])

                print(f'Image: {imageId}, Label: {label}, Model: {model}, Original: {originalPrediction}, Attacked: {attackedPrediction}, Smoothed: {smoothedPrediction}, Noised: {noisedPrediction}, Time: {((end_time - start_time) / 10**9):.2f}s ')
                print(imageId, label, model, originalPrediction, attackedPrediction, smoothedPrediction, noisedPrediction, f'{((end_time - start_time) / 10**9):.2f}', file=results_csv, sep=',')
            results_csv.flush()
            attackedList.append(imageId)
            with open('attacked.csv', 'w') as f:
                print(*attackedList, sep='\n', file=f)
            backend.clear_session()  # Prevent memory leak?

if __name__ == '__main__':

    train_data, train_labels, test_data, test_labels = getCifar10()
    # CNNModel = CNN.getModelCNN()
    # NiNModel = NiN.getModelNiN()
    # VGG16Model = VGG16.getModelVGG16()

    # CNNModel_augmentation = CNN.getModelCNN()
    # NiNModel_augmentation = NiN.getModelNiN()
    # VGG16Model_augmentation = VGG16.getModelVGG16()

    # CNNModel_gaussian = CNN.getModelCNN()
    # NiNModel_gaussian = NiN.getModelNiN()
    # VGG16Model_gaussian = VGG16.getModelVGG16()

    # CNN.fit(CNNModel, train_data, train_labels, test_data, test_labels)
    # saveModel(CNNModel, 'model/CNNModel')

    # NiN.fit(NiNModel, train_data, train_labels, test_data, test_labels)
    # saveModel(NiNModel, 'model/NiNModel')

    # VGG16.fit(VGG16Model, train_data, train_labels, test_data, test_labels)
    # saveModel(VGG16Model, 'model/VGG16Model')

    # fit_augmentataion = ImageAugmentation(x_train=train_data, y_train=train_labels, x_test=test_data, y_test=test_labels)

    # CNNModel_augmentation = fit_augmentataion(CNNModel_augmentation)
    # saveModel(CNNModel_augmentation, 'model/CNNModel_augmented')

    # NiNModel_augmentation = fit_augmentataion(NiNModel_augmentation)
    # saveModel(NiNModel_augmentation, 'model/NiNModel_augmented')

    # VGG16Model_augmentation = fit_augmentataion(VGG16Model_augmentation)
    # saveModel(VGG16Model_augmentation, 'model/VGG16Model_augmented')

    # gaussian_noise_train_data, gaussian_noise_train_labels = GaussianNoise(train_data, train_labels)

    # CNN.fit(CNNModel_gaussian, gaussian_noise_train_data, gaussian_noise_train_labels, test_data, test_labels)
    # saveModel(CNNModel_gaussian, 'model/CNNModel_gaussian')

    # NiN.fit(NiNModel_gaussian, gaussian_noise_train_data, gaussian_noise_train_labels, test_data, test_labels)
    # saveModel(NiNModel_gaussian, 'model/NiNModel_gaussian')

    # VGG16.fit(VGG16Model_gaussian, gaussian_noise_train_data, gaussian_noise_train_labels, test_data, test_labels)
    # saveModel(VGG16Model_gaussian, 'model/VGG16Model_gaussian')

    CNNModel = loadModel('model/CNNModel')
    NiNModel = loadModel('model/NiNModel')
    VGG16Model = loadModel('model/VGG16Model')
    CNNModel_augmentation = loadModel('model/CNNModel_augmented')
    NiNModel_augmentation = loadModel('model/NiNModel_augmented')
    VGG16Model_augmentation = loadModel('model/VGG16Model_augmented')
    CNNModel_gaussian = loadModel('model/CNNModel_gaussian')
    NiNModel_gaussian = loadModel('model/NiNModel_gaussian')
    VGG16Model_gaussian = loadModel('model/VGG16Model_gaussian')

    attack({
        'CNN': CNNModel,
        'NiN': NiNModel,
        'VGG16': VGG16Model,
        'CNN_Augmented': CNNModel_augmentation,
        'NiN_Augmented': NiNModel_augmentation,
        'VGG16_Augmented': VGG16Model_augmentation,
        'CNN_Gaussian': CNNModel_gaussian,
        'NiN_Gaussian': NiNModel_gaussian,
        'VGG16_Gaussian': VGG16Model_gaussian
    }, test_data, test_labels)
