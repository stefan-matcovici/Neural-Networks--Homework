import cPickle, gzip, numpy
import threading

numpy.set_printoptions(threshold=5)


def trunc_data(data, size):
    combined_data = zip(*data)
    combined_data = combined_data[:size]

    return zip(*combined_data)


def translate_data(data, value):
    temp_list = []
    for x in range(len(data[1])):
        temp_list.append(1) if data[1][x] == value else temp_list.append(0)

    return data[0], tuple(temp_list)


def read_data():
    f = gzip.open('../../data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return train_set, valid_set, test_set


def activation(input):
    if input > 0:
        return 1
    return 0


def learn_batch(batch, cls, learning_rate, w):
    delta = 0
    b = 0
    wrong = 0
    for x, t in zip(batch, cls):
        z = numpy.dot(w, x) + b
        output = activation(z)
        delta = numpy.add(delta, numpy.dot(numpy.dot(x, (t - output)), learning_rate))
        b = b + (t - output) * learning_rate
        if t != output:
            wrong += 1
    # print wrong
    return delta, b


def algorithm_batch(data, valid_data):
    w = numpy.random.rand(28 * 28)
    b = numpy.random.rand(1)

    nr_iterations = 10
    learning_rate = 0.1
    batch_size = 10

    while nr_iterations > 0:
        delta = []
        bias = []

        print "iteration: " + str(nr_iterations)
        for i in range(len(data[0]) / batch_size):
            d, b = learn_batch(data[0][i * batch_size:(i + 1) * batch_size],
                               data[1][i * batch_size:(i + 1) * batch_size], learning_rate, w)
            bias.append(b)
            delta.append(d)

        for i in range(len(data[0]) / batch_size):
            w = numpy.add(w, delta[i])
            b = b + bias[i]

        print test_single(w, b, valid_data)
        nr_iterations = nr_iterations - 1

    return w, b


def algorithm(data, valid_data):
    w = numpy.random.rand(28 * 28)
    b = numpy.random.rand(1)

    all_classified = False
    nr_iterations = 5
    learning_rate = 0.1

    while not all_classified and nr_iterations > 0:
        print "iteration: " + str(nr_iterations)
        all_classified = True
        for x, t in zip(*data):
            z = numpy.dot(w, x) + b
            output = activation(z)
            w = numpy.add(w, numpy.dot(numpy.dot(x, (t - output)), learning_rate))
            b = b + (t - output) * learning_rate
            if output != t:
                all_classified = False

        print test_single(w, b, valid_data)
        nr_iterations = nr_iterations - 1

    return w, b


def test_single(w, b, data):
    hit = 0
    for x, t in zip(data[0], data[1]):
        z = numpy.dot(w, x) + b
        output = activation(z)
        if output == t:
            hit = hit + 1

    return hit * 1.0 / len(data[1])


def test_perceptrons(p, data):
    hits = 0
    for x, t in zip(*data):
        max_output = float("-inf")
        hit_perceptron = -1
        for i in range(10):
            z = numpy.dot(p[i][0], x) + p[i][1]
            if z > max_output:
                max_output = z
                hit_perceptron = i

        if hit_perceptron == t:
            hits += 1

    return (hits * 1.0 / len(data[1])) * 100


if __name__ == "__main__":
    train, valid, test = read_data()

    perceptrons = []
    results = []
    perceptron_threads = []
    for i in range(10):
        translated_training_data = translate_data(train, i)
        translated_test_data = translate_data(test, i)
        translated_validation_data = translate_data(valid, i)

        weights, bias = algorithm_batch(translated_training_data, translated_test_data)

        perceptrons.append((weights, bias))

    for p in range(10):
        f = open('p' + str(p), 'w')
        for x in perceptrons[p][0]:
            f.write(str(x) + '\n')
        f.write(str(perceptrons[p][1]) + '\n')

    print test_perceptrons(perceptrons, test)
