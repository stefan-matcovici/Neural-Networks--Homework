import gzip, cPickle, wget
import datetime
import os, sys
import numpy as np
from neuralnetwork import NeuralNetwork

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
LOCAL_DATASET_PATH = "../../data/mnist.pkl.gz"


def shuffle(data):
    zipped_data = zip(*data)
    np.random.shuffle(zipped_data)

    return zip(*zipped_data)


def translate_output(value):
    e = np.zeros((10, 1))
    e[value] = 1.0
    return e


def get_dataset():
    """Downloads the archive with the dataset if not present at the specified path.
       Opens the archive and returns training set, validation set and testing set as a tuple
    """

    if not os.path.isfile(LOCAL_DATASET_PATH) or not os.access(LOCAL_DATASET_PATH, os.R_OK):
        wget.download(MNIST_URL, LOCAL_DATASET_PATH)

    f = gzip.open(LOCAL_DATASET_PATH, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    training_inputs = [np.reshape(x, (784, 1)) for x in train_set[0]]
    training_results = [translate_output(y) for y in train_set[1]]
    training_set = (training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in valid_set[0]]
    validation_set = (validation_inputs, valid_set[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in test_set[0]]
    test_set = (test_inputs, test_set[1])

    return training_set, validation_set, test_set
