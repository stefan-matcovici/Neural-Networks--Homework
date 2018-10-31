import cPickle, gzip, os, sys
import numpy as np
import wget
import scipy.misc
import matplotlib.pyplot as plt

import logging
import argparse
import math

from repository import get_dataset
from neuralnetwork import NeuralNetwork

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def trim_dataset(dataset, size):
    return [dataset[0][:size], dataset[1][:size]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train simple perceptrons to recognize handwritten digits.')
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate')
    parser.add_argument('-ni', '--no-iterations', type=int, help='no of iterations')
    parser.add_argument('-bs', '--batch-size', type=int, help='batch size')
    parser.add_argument('-t', '--trim', type=int, help='trim data set to size', default=0)
    parser.add_argument('--save', help='save the model in a file', action="store_const", const=True)
    parser.add_argument('--load', type=str, help='load the model from a file')

    parse_result = parser.parse_args(sys.argv[1:])

    train_set, valid_set, test_set = get_dataset()
    # # plt.imshow(train_set[0][0].reshape((28, 28)), interpolation='nearest')
    # # plt.show()
    parse_result.trim = 50000

    if parse_result.trim != 0:
        train_set = trim_dataset(train_set, parse_result.trim)
        valid_set = trim_dataset(valid_set, parse_result.trim)
        test_set = trim_dataset(test_set, parse_result.trim)

    nnetwork = NeuralNetwork((784, 36, 10))
    # for layer in nnetwork.weights:
    #     print(layer.shape)
    # for bias in nnetwork.biases:
    #     print(bias.shape)

    # nnetwork.train(train_set, valid_set, parse_result.learning_rate, parse_result.no_iterations, parse_result.batch_size)
    nnetwork.train(train_set, valid_set, 0.5, 10, 10)
    nnetwork.test(test_set)

    # print(train_set[0][0].shape)
    # if parse_result.load:
    #     classifier = load(parse_result.load)
    # else:
    #     if parse_result.batch_size is not None:
    #         classifier.train_in_batches(train_set, valid_set, parse_result.learning_rate, parse_result.no_iterations, parse_result.batch_size, parse_result.adaline)
    #     else:
    #         classifier.train(train_set, valid_set, parse_result.learning_rate, parse_result.no_iterations, parse_result.adaline)

    #     if parse_result.save:
    #         save("model-"+str(abs(int(math.log10(0.001))))+"-"+ str(parse_result.no_iterations) + (("-"+str(parse_result.batch_size)) if parse_result.batch_size else "") + ("-adaline" if parse_result.adaline else ""), classifier)

    # classifier.test(test_set)
