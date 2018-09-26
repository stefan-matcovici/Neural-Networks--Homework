import cPickle, gzip, os
import numpy as np
import wget
import scipy.misc
import matplotlib.pyplot as plt

from perceptron import Perceptron
from classifier import Classifier

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
LOCAL_DATASET_PATH = "../data/mnist.pkl.gz"

def get_dataset(path):
    """Downloads the archive with the dataset if not present at the specified path.
       Opens the archive and returns training set, validation set and testing set as a tuple
    """

    if not os.path.isfile(path) or not os.access(path, os.R_OK):
        wget.download(MNIST_URL, path)

    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
         
    return train_set, valid_set, test_set

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional

if __name__ == "__main__":
    train_set, valid_set, test_set = get_dataset(LOCAL_DATASET_PATH)
    # plt.imshow(train_set[0][0].reshape((28, 28)), interpolation='nearest')
    # plt.show()
    train_set = [train_set[0][:100], train_set[1][:100]]
    test_set = [test_set[0][:100], test_set[1][:100]]

    perceptron = Perceptron(784, 1)
    perceptron.train(train_set, 0.001, 10)
    perceptron.test(test_set)

    # classifier = Classifier()
    # classifier.train(train_set, 0.0001, 10)
    # classifier.test(test_set)

    