# ignore: E501
import pickle
import numpy as np


def unpickle(file: str):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def getCifar10():
    # Start by loading training data
    train_filenames = (f'data_batch_{i}' for i in range(1, 6))
    train_data = {}
    for filename in train_filenames:
        loaded_data = unpickle(f'datasets/cifar-10-original/{filename}')
        if 'data' in train_data:
            train_data['data'] = np.concatenate((train_data['data'], loaded_data[b'data']), axis=0)
            train_data['labels'] += loaded_data[b'labels']
        else:
            train_data['data'] = loaded_data[b'data']
            train_data['labels'] = loaded_data[b'labels']

    # Then load test data
    test_filename = 'test_batch'
    test_data = {}
    loaded_data = unpickle(f'datasets/cifar-10-original/{test_filename}')
    test_data['data'] = loaded_data[b'data']
    test_data['labels'] = loaded_data[b'labels']

    # Return both
    return {'train_data': train_data, 'test_data': test_data}


if __name__ == '__main__':
    data = getCifar10()
