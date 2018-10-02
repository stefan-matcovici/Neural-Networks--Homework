import cPickle, gzip, os, sys
import numpy as np
import wget
import scipy.misc
import matplotlib.pyplot as plt

from perceptron import Perceptron
from classifier import Classifier
from repository import *

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
    train_set, valid_set, test_set = get_dataset()
    # plt.imshow(train_set[0][0].reshape((28, 28)), interpolation='nearest')
    # plt.show()
    # train_set = [train_set[0][:10], train_set[1][:10]]
    # test_set = [test_set[0][:100], test_set[1][:100]]

    # p = Perceptron(784, 7)
    # p.train(train_set, 0.1, 5)
    # p.test(test_set)

    # perceptron.load("test")
    # perceptron.test(test_set)

    classifier = Classifier()
    # classifier.train(train_set, 0.001, 5)
    # save("working", classifier)
    classifier = load("working21618")
    classifier.test(test_set)

    # print(test_set[1][0])
    # print(p.get_result(test_set[0][0]))
    
    # classifier.test(test_set)


    