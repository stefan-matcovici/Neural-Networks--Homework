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


def get_dataset():
    """Downloads the archive with the dataset if not present at the specified path.
       Opens the archive and returns training set, validation set and testing set as a tuple
    """

    if not os.path.isfile(LOCAL_DATASET_PATH) or not os.access(LOCAL_DATASET_PATH, os.R_OK):
        wget.download(MNIST_URL, LOCAL_DATASET_PATH)

    f = gzip.open(LOCAL_DATASET_PATH, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return shuffle(train_set), valid_set, test_set
