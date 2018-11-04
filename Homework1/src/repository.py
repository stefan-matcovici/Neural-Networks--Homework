import gzip, cPickle, wget
import datetime
import os, sys
import numpy as np
from classifier import Classifier

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
LOCAL_DATASET_PATH = "../data/mnist.pkl.gz"


def shuffle(data):
    zipped_data = zip(*data)
    np.random.shuffle(zipped_data)

    return zip(*zipped_data)


def translate_data(data):
    temp = []
    for value in data[1]:
        t = np.zeros(10)
        t[value] = 1
        temp.append(t)

    return np.reshape(data[0], newshape=(len(data[0]), 784, 1)), np.reshape(np.array(temp),
                                                                            newshape=(len(data[1]), 10, 1))


def read_data():
    f = gzip.open('../../data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return translate_data(train_set), translate_data(valid_set), translate_data(test_set)


def save(model_name, classifier):
    now = datetime.datetime.now()
    data = np.array([])

    with open(os.path.join("models", model_name + "-" + str(now.day) + "-" + str(now.hour) + "-" + str(now.minute)),
              'wb') as output:
        for perceptron in classifier.perceptrons:
            data = np.append(data, [perceptron.weights, perceptron.bias])
        cPickle.dump(data, output, cPickle.HIGHEST_PROTOCOL)


def load(file_name):
    classifier = Classifier()
    with open("models/" + file_name, 'rb') as input:
        data = cPickle.load(input)
        i = 0
        for perceptron in classifier.perceptrons:
            perceptron.weights = data[i]
            perceptron.bias = data[i + 1]
            i += 2
    return classifier
