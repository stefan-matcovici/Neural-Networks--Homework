import argparse
import logging
import sys

from neuralnetwork import NeuralNetwork
from repository import get_dataset

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
    parser.add_argument('-r', '--regularization', type=float, help='regularization lambda')
    parser.add_argument('-m', '--momentum', type=float, help='momentum coeficient')
    parser.add_argument('-t', '--trim', type=int, help='trim data set to size', default=0)

    parse_result = parser.parse_args(sys.argv[1:])

    train_set, valid_set, test_set = get_dataset()

    if parse_result.trim is not None and parse_result.trim != 0:
        train_set = trim_dataset(train_set, parse_result.trim)
        valid_set = trim_dataset(valid_set, parse_result.trim)
        test_set = trim_dataset(test_set, parse_result.trim)

    nnetwork = NeuralNetwork((784, 100, 10))
    nnetwork.train(train_set, valid_set, parse_result.learning_rate, parse_result.no_iterations,
                   parse_result.batch_size, parse_result.regularization, parse_result.momentum)
    # nnetwork.train(train_set, valid_set, 0.1, 10, 10, 0.1, 0.9)
    print("Test accuracy: %f" % nnetwork.test(test_set))

