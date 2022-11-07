# ignore: E501
import pickle
import numpy as np
from keras.utils.np_utils import to_categorical


def unpickle(file: str):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def getCifar10():
    # Start by loading training data
    train_filenames = (f'data_batch_{i}' for i in range(1, 6))
    train_data = {}
    for filename in train_filenames:
        loaded_data = unpickle(f'Project/datasets/cifar-10-original/{filename}')
        if 'data' in train_data:
            train_data['data'] = np.concatenate((train_data['data'], loaded_data[b'data']), axis=0)
            train_data['labels'] += loaded_data[b'labels']
        else:
            train_data['data'] = loaded_data[b'data']
            train_data['labels'] = loaded_data[b'labels']

    # Then load test data
    test_filename = 'test_batch'
    test_data = {}
    loaded_data = unpickle(f'Project/datasets/cifar-10-original/{test_filename}')
    test_data['data'] = loaded_data[b'data']
    test_data['labels'] = loaded_data[b'labels']

    # Return both
    return {'train_data': train_data, 'test_data': test_data}


def preprocess_data(train_data_dict, test_data_dict):
    train_labels = train_data_dict['labels']
    train_data = train_data_dict['data']
    test_labels = test_data_dict['labels']
    test_data = test_data_dict['data']
    train_data = train_data / 255.
    test_data = test_data / 255.
    train_data = np.reshape(train_data, (len(train_data), 3, 32, 32))
    train_data = np.moveaxis(train_data, 1, -1)
    test_data = np.reshape(test_data, (len(test_data), 3, 32, 32))
    test_data = np.moveaxis(test_data, 1, -1)
    train_labels = np.array(train_labels)
    train_labels = to_categorical(train_labels, 10)
    test_labels = np.array(test_labels)
    test_labels = to_categorical(test_labels, 10)
    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    data = getCifar10()
