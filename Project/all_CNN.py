import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models
import data as d

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']





if __name__ == "__main__":
    data = d.getCifar10()
    train_data = data['train_data']['data']
    train_labels = data['train_data']['labels']
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_data[i])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()