import cPickle
import gzip
import numpy as np


def translate_output(value):
    e = np.zeros((10, 1))
    e[value] = 1.0
    return e


class DataLoader(object):
    def __init__(self):
        self.training_data = None
        self.validation_data = None
        self.test_data = None

    def __load_data__(self):
        f = gzip.open('mnist.pkl.gz', 'rb')
        self.training_data, self.validation_data, self.test_data = cPickle.load(f)
        f.close()

    def get_data(self):
        self.__load_data__()

        training_inputs = [np.reshape(x, (784, 1)) for x in self.training_data[0]]
        training_results = [translate_output(y) for y in self.training_data[1]]
        training_set = zip(training_inputs, training_results)

        validation_inputs = [np.reshape(x, (784, 1)) for x in self.validation_data[0]]
        validation_set = zip(validation_inputs, self.validation_data[1])

        test_inputs = [np.reshape(x, (784, 1)) for x in self.test_data[0]]
        test_set = zip(test_inputs, self.test_data[1])
        return training_set, validation_set, test_set

