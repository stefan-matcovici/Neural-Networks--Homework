import numpy as np


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

    def train(self, training_set, learning_rate, no_iterations, batch_size=1, adaline=False):
        all_classified = False
        while not all_classified and no_iterations > 0:
            all_classified = True
            for example, target in zip(training_set[0], np.array([1 if target == self.target_number else 0 for target in training_set[1]])):
                z = np.dot(self.weights, example) + self.bias

                correction = 0
                if adaline:
                    correction = z
                else:
                    correction = self.__activation__(z)

                self.weights = np.add(
                    self.weights, (target-correction)*learning_rate*self.weights)
                self.bias = self.bias + (target-correction)*learning_rate

                if correction != target:
                    all_classified = False
            no_iterations -= 1
    

    def test(self, testing_set):
        correct = 0

        for example, target in zip(testing_set[0], np.array([1 if target == self.target_number else 0 for target in testing_set[1]])):
            z = np.dot(self.weights, example) + self.bias
            correction = self.__activation__(z)

            if correction == target:
                correct += 1
        
        print((float(correct)/float(len(testing_set[0])))*100)

