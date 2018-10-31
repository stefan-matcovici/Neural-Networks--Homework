import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork(object):

    def __init__(self, layers_sizes):
        self.no_layers = len(layers_sizes) - 1
        self.layers_sizes = layers_sizes
        self.__initialize_model__(layers_sizes)

    def __sigmoid__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __sigmoid_derivative__(self, x):
        return np.multiply(x, np.subtract(np.ones(x.shape[0]), x))

    def __mse_error__(self, data):
        error = 0
        for sample, target in zip(*data):
            outputs = self.__feedforward__(sample)
            error += np.sum((outputs[-1] - target) ** 2)

        return error / (2 * len(data[0]))

    def __initialize_model__(self, layers_sizes):
        self.weights = []
        self.biases = []
        for i in range(1, len(layers_sizes)):
            self.weights.append(
                np.random.normal(0, 1.0 / np.sqrt(layers_sizes[i]), (layers_sizes[i - 1], layers_sizes[i])))
            self.biases.append(np.random.randn(layers_sizes[i], ))

    def __transform_target__(self, target):
        result = np.zeros(10)
        result[int(target)] = 1.0

        return result

    def __split_into_batches__(self, training_set, batch_size):
        sample_batches = np.split(training_set[0], range(batch_size, len(training_set[0]), batch_size))
        sample_targets = np.split(training_set[1], range(batch_size, len(training_set[1]), batch_size))

        return zip(sample_batches, sample_targets)

    def __feedforward__(self, samples):
        outputs = [samples]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            input = outputs[i]
            outputs.append(self.__sigmoid__(np.dot(input, self.weights[i]) + self.biases[i]))
        return outputs

    def __train_batch__(self, batch, learning_rate):
        weights_adjustements = [None for i in range(self.no_layers)]
        bias_adjustments = [None for i in range(self.no_layers)]
        for i, (sample, target) in enumerate(zip(*batch)):
            self.errors = []

            outputs = self.__feedforward__(sample)

            # error computation
            for layer in range(self.no_layers, 0, -1):
                y = outputs[layer]
                if layer == self.no_layers:
                    # last layer
                    error = np.multiply(self.__sigmoid_derivative__(y), np.subtract(y, target))
                else:
                    error = np.multiply(self.__sigmoid_derivative__(y), (np.dot(self.weights[layer], error)))

                if i == 0:
                    # first pass, initialize adjustements
                    weights_adjustements[layer - 1] = np.dot(outputs[layer - 1][:, np.newaxis],
                                                             error.reshape(1, error.shape[0]))
                    bias_adjustments[layer - 1] = np.copy(error)
                else:
                    weights_adjustements[layer - 1] = np.add(weights_adjustements[layer - 1],
                                                             np.dot(outputs[layer - 1][:, np.newaxis],
                                                                    error.reshape(1, error.shape[0])))
                    bias_adjustments[layer - 1] = np.add(bias_adjustments[layer - 1], error)

        return weights_adjustements, bias_adjustments

    def train(self, training_set, valid_set, learning_rate, no_iterations, batch_size):
        training_set[1] = list(map(lambda x: self.__transform_target__(x), training_set[1]))
        # training_set[0] = list(map(lambda x: x[:, np.newaxis], training_set[0]))

        errors = []

        for i in range(no_iterations):
            batches = self.__split_into_batches__(training_set, batch_size)

            for batch in batches:
                weight_adjustements, bias_adjustements = self.__train_batch__(batch, learning_rate)
                for j, (w, adjust) in enumerate(zip(self.weights, weight_adjustements)):
                    self.weights[j] = np.add(w, -(learning_rate / batch_size) * adjust)

                for j, (b, adjust) in enumerate(zip(self.biases, bias_adjustements)):
                    self.biases[j] = np.add(b, -(learning_rate / batch_size) * adjust)

            error = self.__mse_error__(training_set)
            errors.append(error)
            print(error)
            plt.ylim(0, 140)
            plt.xlim(0, 10)
            plt.plot(errors)
            plt.show()

    def test(self, test_set):
        right = 0
        for sample, target in zip(*test_set):
            outputs = self.__feedforward__(sample)
            prediction = np.argmax(outputs[-1])
            if prediction != target:
                right += 1
        print(right)
        print(((1.0*right)/(1.0*len(test_set[0])))*100)
