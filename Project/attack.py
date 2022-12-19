from DE import perturbImage
import NiN
import CNN
import VGG16
from data import getCifar10, saveModel
import numpy as np
import matplotlib.pyplot as plt
from keras import models


def attack(models: "dict[str, models.Model]", test_data: "np.ndarray", test_labels: "np.ndarray"):
    for imageId in np.random.choice(np.arange(len(test_data)), size=100, replace=False):
        image = test_data[imageId]
        label = np.argmax(test_labels[imageId])

        plt.imsave(f'images/original/{imageId}_label_{label}.png', image)
        np.save(f'data/original/{imageId}', image)

        for model in models:
            originalPrediction = np.argmax(models[model].predict(np.array([image])))

            attacked = perturbImage(image, label, models[model])

            plt.imsave(f'images/{model}/{imageId}_label_{label}.png', attacked)
            np.save(f'data/{model}/{imageId}.np', attacked)

            attackedPrediction = np.argmax(models[model].predict(np.array([attacked])))

            print(f'Image: {imageId}, Model: {model}, Original: {originalPrediction}, Attacked: {attackedPrediction}')


if __name__ == '__main__':
    train_data, train_lables, test_data, test_labels = getCifar10()
    CNNModel = CNN.getModelCNN()
    NiNModel = NiN.getModelNiN()
    VGG16Model = VGG16.getModelVGG16()

    CNN.fit(CNNModel, train_data, train_lables, test_data, test_data)
    saveModel(CNNModel, 'model/CNNModel')

    NiN.fit(NiNModel, train_data, train_lables, test_data, test_data)
    saveModel(CNNModel, 'model/NiNModel')

    VGG16.fit(VGG16Model, train_data, train_lables, test_data, test_data)
    saveModel(CNNModel, 'model/VGG16Model')

    attack({'CNN': CNNModel, 'NiN': NiNModel, 'VGG16': VGG16Model}, test_data, test_labels)
