import numpy as np
from timeit import default_timer as timer


class NeuralNetwork(object):

    def __init__(self, layers_sizes):
        self.no_layers = len(layers_sizes) - 1
        self.layers_sizes = layers_sizes
        self.__initialize_model__(layers_sizes)

    def __sigmoid__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __softmax__(self, x):
        all_sum = np.sum(np.exp(x))
        return x / all_sum

    def __sigmoid_derivative__(self, x):
        return x * (1 - x)

    def __mse_error__(self, data):
        error = 0
        for sample, target in zip(*data):
            outputs = self.__feedforward__(sample)
            error += np.sum((outputs[-1] - target) ** 2)

        return error / (2 * len(data[0]))

    def __cross_entropy_error__(self, data):
        error = 0
        for sample, target in zip(*data):
            outputs = self.__feedforward__(sample)
            y = outputs[-1]
            error += np.sum(target * np.log(y) + (1 - target) * np.log(1 - y))

        return -error / (len(data[0]))

    def __initialize_model__(self, layers_sizes):
        self.weights = []
        self.biases = []

        self.weights = np.array([np.random.randn(y, x) / np.sqrt(x)
                                 for x, y in zip(self.layers_sizes[:-1], self.layers_sizes[1:])])
        self.biases = np.array([np.random.randn(y, 1) for y in self.layers_sizes[1:]])

    def __split_into_batches__(self, training_set, batch_size):
        sample_batches = np.split(training_set[0], range(batch_size, len(training_set[0]), batch_size))
        sample_targets = np.split(training_set[1], range(batch_size, len(training_set[1]), batch_size))

        return zip(sample_batches, sample_targets)

    def __feedforward__(self, x):
        outputs = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            input = outputs[i]
            if i != len(self.layers_sizes):
                outputs.append(self.__sigmoid__(np.dot(self.weights[i], input) + self.biases[i]))
            else:
                outputs.append(self.__softmax__(np.dot(self.weights[i], input) + self.biases[i]))
        return outputs

    def __train_batch__(self, batch, learning_rate):
        weights_adjustements = np.array([np.zeros(shape=w.shape) for w in self.weights])
        bias_adjustments = np.array([np.zeros(shape=b.shape) for b in self.biases])
        for i, (sample, target) in enumerate(zip(*batch)):
            self.errors = []

            outputs = self.__feedforward__(sample)

            # error computation
            for layer in range(self.no_layers, 0, -1):
                y = outputs[layer]
                if layer == self.no_layers:
                    # last layer
                    error = y - target
                else:
                    error = self.__sigmoid_derivative__(y) * np.transpose(np.dot(error.T, self.weights[layer]))

                weights_adjustements[layer - 1] += np.dot(error, outputs[layer - 1].T)
                bias_adjustments[layer - 1] += error

        return weights_adjustements, bias_adjustments

    def train(self, training_set, valid_set, learning_rate, no_iterations, batch_size, l, friction):

        n = len(training_set[0])

        for i in range(no_iterations):
            start = timer()
            batches = self.__split_into_batches__(training_set, batch_size)

            last_adjust = None
            for batch in batches:
                weight_adjustements, bias_adjustements = self.__train_batch__(batch, learning_rate)

                if last_adjust is not None:
                    weight_adjustements = friction * last_adjust - (learning_rate / batch_size) * weight_adjustements
                else:
                    weight_adjustements = (learning_rate / batch_size) * weight_adjustements
                last_adjust = np.copy(weight_adjustements)

                self.weights = (1 - learning_rate * l / n) * self.weights + weight_adjustements
                self.biases = self.biases - (learning_rate / batch_size) * bias_adjustements

            end = timer()
            print("End of iteration %d which took %.2f seconds" % (i, end - start))
            print("Cross entropy error: %f" % self.__cross_entropy_error__(training_set))
            print("Validation accuracy: %f" % self.test(valid_set))

    def test(self, test_set):
        right = 0
        for sample, target in zip(*test_set):
            outputs = self.__feedforward__(sample)
            prediction = np.argmax(outputs[-1])
            if prediction == target:
                right += 1
        return ((1.0 * right) / (1.0 * len(test_set[0]))) * 100
