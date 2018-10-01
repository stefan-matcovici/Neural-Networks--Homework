from perceptron import Perceptron
import numpy as np
import datetime
import cPickle

class Classifier(object):
    def __init__(self):
        self.perceptrons = []
        for i in range(10):
            self.perceptrons.append(Perceptron(784, i))
    
    def train(self, training_set, learning_rate, no_iterations, adaline=False):
        for perceptron in self.perceptrons:
            perceptron.train(training_set, learning_rate, no_iterations, adaline)
    
    def train_in_batches(self, training_set, learning_rate, no_iterations, no_batches, adaline=False):
        for perceptron in self.perceptrons:
            perceptron.train_in_batches(training_set, learning_rate, no_iterations, no_batches, adaline)
    
    def predict(self, sample):
        return np.argmax([perceptron.predict(sample) for perceptron in self.perceptrons])
    
    def get_result(self, sample):
        return np.argmax([perceptron.get_result(sample) for perceptron in self.perceptrons])

    def __get_target_array__(self, input_array, target_number):
        return np.array([1 if target == target_number else 0 for target in input_array])

    def test(self, testing_set):
        correct = 0

        for example, target in zip(testing_set[0], testing_set[1]):
            output = self.get_result(example)

            if output == target:
                correct += 1

        print((float(correct)/float(len(testing_set[1])))*100)

