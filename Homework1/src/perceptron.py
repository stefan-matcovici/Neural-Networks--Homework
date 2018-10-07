import numpy as np
import logging

logger = logging.getLogger(__name__)

class Perceptron(object):
    def __init__(self, input_size, target_number):
        self.target_number = target_number
        self.__initialize_model__(input_size)

    def __initialize_model__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = 0

    def __activation__(self, input):
        if input > 0:
            return 1
        return 0

    def __get_target_array__(self, input_array):
        return np.array([1 if target == self.target_number else 0 for target in input_array])

    def __split_into_batches__(self, training_set, no_batches):
        batch_size = len(training_set[0])/no_batches
        sample_batches = np.split(training_set[0], range(len(training_set[0]), 0, batch_size))
        sample_targets = np.split(training_set[1], range(len(training_set[1]), 0, batch_size))

        return zip(sample_batches, sample_targets)

    def __train_episode__(self, samples, targets, learning_rate, adaline):
        weight_adjustements = np.zeros(samples[0].shape)
        bias_adjustement = 0
        for example, target in zip(samples, self.__get_target_array__(targets)):
            z = np.dot(self.weights, example) + self.bias

            output = 0
            if adaline:
                output = z
            else:
                output = self.__activation__(z)

            weight_adjustements = np.add(
                weight_adjustements, (target-output)*learning_rate*self.weights)
            bias_adjustement = bias_adjustement + (target-output)*learning_rate
        return weight_adjustements, bias_adjustement
            
    def train(self, training_set, validation_set, learning_rate, no_iterations, adaline=False):
        all_classified = False
    
        logger.info("Parameters: learning_rate=%f, no_iterations=%d, adaline=%s", learning_rate, no_iterations, adaline)

        while not all_classified and no_iterations > 0:
            all_classified = True
            
            for example, target in zip(training_set[0], self.__get_target_array__(training_set[1])):
                z = np.dot(self.weights, example) + self.bias

                output = 0
                if adaline:
                    output = z
                else:
                    output = self.__activation__(z)

                self.weights = np.add(
                    self.weights, (target-output)*learning_rate*example)
                self.bias = self.bias + (target-output)*learning_rate

                if output != target:
                    all_classified = False
            logger.info("Iteration %d accuracy on validation set: %f", no_iterations, self.test(validation_set))
            no_iterations -= 1

    def train_in_batches(self, training_set, learning_rate, no_iterations, no_batches, adaline=False):
        training_data = self.__split_into_batches__(training_set, no_batches)
        batch_adjustements = []
        while no_iterations > 0:
            for batch in training_data:
                weight_adjustements, bias_adjustement = self.__train_episode__(batch[0], batch[1], learning_rate, adaline)
                batch_adjustements.append((weight_adjustements, bias_adjustement))
            for batch_adjustement in batch_adjustements:
                self.weights = np.add(self.weights, batch_adjustement[0])
                self.bias = self.bias + batch_adjustement[1]
            no_iterations -= 1   

    def predict(self, sample):
        return self.__activation__(np.dot(self.weights, sample) + self.bias)

    def get_result(self, sample):
        return np.dot(self.weights, sample) + self.bias

    def test(self, testing_set):
        correct = 0

        for example, target in zip(testing_set[0], self.__get_target_array__(testing_set[1])):
            output = self.predict(example)

            if output == target:
                correct += 1

        return ((float(correct)/float(len(testing_set[1])))*100)
