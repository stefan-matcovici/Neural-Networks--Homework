import numpy, random


def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))


def softmax(x):
    total = numpy.sum(numpy.exp(x))
    return numpy.exp(x) / total


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_weights = numpy.random.normal(0, 1.0 / numpy.sqrt(input_size), (hidden_size, input_size))
        self.hidden_biases = numpy.random.randn(hidden_size, 1)

        self.output_weights = numpy.random.normal(0, 1.0 / numpy.sqrt(hidden_size), (output_size, hidden_size))
        self.output_biases = numpy.random.randn(output_size, 1)

        self.previous_hidden_weights_adjustments = numpy.zeros(self.input_size * self.hidden_size,
                                                               dtype="float64").reshape(
            (self.hidden_size, self.input_size))

        self.previous_output_weights_adjustments = numpy.zeros(self.hidden_size * self.output_size,
                                                               dtype="float64").reshape(
            (self.output_size, self.hidden_size))

    def train(self, training_data, iterations, learning_grade, batch_size, l, momentum, validation_data=None,
              training_accuracy=False):
        validation_data_size = len(validation_data)
        training_data_size = len(training_data)

        print "Neural network with parameters:"
        print "Iterations: {}".format(iterations)
        print "Learning grade: {}".format(learning_grade)
        print "Batch_size: {}".format(batch_size)
        print "Lambda: {}".format(l)
        print "Momentum: {}".format(momentum)

        for i in xrange(iterations):
            random.shuffle(training_data)

            batches = [
                training_data[k:k + batch_size]
                for k in xrange(0, training_data_size, batch_size)]

            for batch in batches:
                self.train_batch(batch, training_data_size, learning_grade, l, momentum)

            print "Iteration {} training complete".format(i)
            if validation_data:
                accuracy = self.accuracy(validation_data)
                print "Accuracy on validation data: {} / {}".format(
                    accuracy, validation_data_size)

            if training_accuracy:
                accuracy = self.accuracy(training_data, vectorized=True)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, training_data_size)

    def train_batch(self, batch, training_data_size, learning_grade, l, momentum):
        hidden_weights_adjustments = numpy.zeros(self.input_size * self.hidden_size,
                                                 dtype="float64").reshape(
            (self.hidden_size, self.input_size))

        output_weights_adjustments = numpy.zeros(self.hidden_size * self.output_size,
                                                 dtype="float64").reshape(
            (self.output_size, self.hidden_size))

        hidden_bias_adjustments = numpy.zeros(self.hidden_size, dtype="float64").reshape(
            self.hidden_size, 1)
        output_bias_adjustments = numpy.zeros(self.output_size, dtype="float64").reshape(
            self.output_size, 1)
        batch_size = len(batch)

        for x, y in batch:
            # forward pass
            outputs = []
            z = []
            # hidden layer
            r = numpy.dot(self.hidden_weights, x) + self.hidden_biases
            z.append(r)
            previous_output = sigmoid(r)
            outputs.append(previous_output)

            # output layer
            r = numpy.dot(self.output_weights, outputs[0]) + self.output_biases
            z.append(r)
            previous_output = softmax(r)
            outputs.append(previous_output)

            # output layer
            delta = outputs[1] - y

            output_bias_adjustments = numpy.add(output_bias_adjustments, delta)
            output_weights_adjustments = numpy.add(output_weights_adjustments, numpy.dot(delta, outputs[0].transpose()))

            # hidden layer
            sp = sigmoid_prime(z[0])
            delta = numpy.dot(self.output_weights.transpose(), delta) * sp

            hidden_bias_adjustments = numpy.add(hidden_bias_adjustments, delta)
            hidden_weights_adjustments = numpy.add(hidden_weights_adjustments,
                                                   numpy.dot(delta, x.transpose()))

        # momentum
        hidden_weights_adjustments = self.previous_hidden_weights_adjustments * momentum - numpy.multiply(
            learning_grade / batch_size, hidden_weights_adjustments)
        output_weights_adjustments = self.previous_output_weights_adjustments * momentum - numpy.multiply(
            learning_grade / batch_size, output_weights_adjustments)

        self.previous_hidden_weights_adjustments = hidden_weights_adjustments
        self.previous_output_weights_adjustments = output_weights_adjustments

        # adjust weights
        self.hidden_weights = numpy.add(
            numpy.multiply((1 - learning_grade * l / training_data_size), self.hidden_weights),
            hidden_weights_adjustments)
        self.output_weights = numpy.add(
            numpy.multiply((1 - learning_grade * l / training_data_size), self.output_weights),
            output_weights_adjustments)

        self.hidden_biases = numpy.subtract(self.hidden_biases,
                                            numpy.multiply(learning_grade / batch_size, hidden_bias_adjustments))
        self.output_biases = numpy.subtract(self.output_biases,
                                            numpy.multiply(learning_grade / batch_size, output_bias_adjustments))

    def fast_forward_pass(self, inputs):
        inputs = sigmoid(numpy.dot(self.hidden_weights, inputs) + self.hidden_biases)
        inputs = softmax(numpy.dot(self.output_weights, inputs) + self.output_biases)

        return inputs

    def accuracy(self, data, vectorized=False):
        if vectorized:
            output = [numpy.argmax(y) for x, y in data]
        else:
            output = [x[1] for x in data]
        results = [(numpy.argmax(self.fast_forward_pass(x[0])), y)
                   for x, y in zip(data, output)]
        return numpy.sum([int(x == y) for (x, y) in results])
