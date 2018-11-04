from __future__ import division
import logging

import numpy as np

logger = logging.getLogger(__name__)


class Classifier(object):
    def __init__(self):
        self.weights = np.zeros(shape=(28 * 28, 10))
        self.biases = np.zeros(shape=(10, 1))

    def activation(self, x):
        t = np.zeros(10)
        for i in range(len(x)):
            if x[i] > 0:
                t[i] = 1
            else:
                t[i] = 0
        return np.reshape(np.array(t), newshape=(10, 1))

    def train(self, training_set, validation_set, learning_rate, no_iterations, adaline=False):
        logger.info("Parameters: learning_rate=%f, no_iterations=%d, adaline=%s", learning_rate, no_iterations, adaline)
        all_classified = False

        while not all_classified and no_iterations > 0:
            print "iteration: " + str(no_iterations)
            all_classified = True
            for x, t in zip(*training_set):
                z = np.dot(x.T, self.weights).T + self.biases

                if adaline:
                    output = z
                else:
                    output = self.activation(z)

                self.weights = self.weights + np.dot(x, (t - output).T) * learning_rate
                self.biases = self.biases + (t - output) * learning_rate
                if np.argmax(output) != np.argmax(t):
                    all_classified = False

            logger.info("Iteration %d accuracy on validation set: %f", no_iterations, self.test(validation_set))
            no_iterations = no_iterations - 1

    def train_in_batches(self, training_set, validation_set, learning_rate, no_iterations, batch_size, adaline=False):
        training_data = self.__split_into_batches__(training_set, batch_size)
        while no_iterations > 0:
            for batch in training_data:
                weight_adjustements, bias_adjustement = self.__train_episode__(batch[0], batch[1], learning_rate,
                                                                               adaline)
                self.weights += weight_adjustements
                self.biases += bias_adjustement
            logger.info("Iteration %d accuracy on validation set: %f", no_iterations, self.test(validation_set))
            no_iterations -= 1

    def __split_into_batches__(self, training_set, batch_size):
        sample_batches = np.split(training_set[0], range(batch_size, len(training_set[0]), batch_size))
        sample_targets = np.split(training_set[1], range(batch_size, len(training_set[1]), batch_size))

        return zip(sample_batches, sample_targets)

    def test(self, data):
        right = 0

        for x, t in zip(*data):
            z = np.dot(x.T, self.weights).T + self.biases
            output = np.argmax(z)
            if output == np.argmax(t):
                right += 1

        return (right / len(data[0])) * 100

    def __train_episode__(self, samples, targets, learning_rate, adaline):
        weight_adjustements = np.zeros(self.weights.shape)
        bias_adjustements = np.zeros(self.biases.shape)
        for x, t in zip(samples, targets):
            z = np.dot(x.T, self.weights).T + self.biases

            if adaline:
                output = z
            else:
                output = self.activation(z)

            weight_adjustements += np.dot(x, (t - output).T) * learning_rate
            bias_adjustements += (t - output) * learning_rate

        return weight_adjustements, bias_adjustements
