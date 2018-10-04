import cPickle, gzip, os, sys
import numpy as np
import wget
import scipy.misc
import matplotlib.pyplot as plt

from perceptron import Perceptron
from classifier import Classifier
from repository import *

import logging

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional

def trim_dataset(dataset, size):
    return [dataset[0][:size], dataset[1][:size]]

if __name__ == "__main__":
    train_set, valid_set, test_set = get_dataset()
    # plt.imshow(train_set[0][0].reshape((28, 28)), interpolation='nearest')
    # plt.show()

    classifier = Classifier()
    classifier.train(train_set, valid_set, 0.001, 10)
    # save("working", classifier)
    # classifier = load("working21618")
    # classifier.test(test_set)


    