import numpy
import cPickle, gzip
from DataLoader import DataLoader
from Network import Network


def trunc_data(data, size):
    combined_data = zip(*data)
    combined_data = combined_data[:size]

    return zip(*combined_data)


def translate_data(data):
    temp_list = []
    for x in range(len(data[1])):
        temp_list2 = []
        for i in range(10):
            temp_list2.append(1 if i == data[1][x] else 0)
        temp_list.append(temp_list2)

    return data[0], tuple(temp_list)


if __name__ == "__main__":

    # train_set = (numpy.array([[2, 6]], dtype="float32"), numpy.array([0], dtype="int64"))
    # valid_set = (numpy.array([[2, 6]], dtype="float32"), numpy.array([0], dtype="int64"))

    # neuronal_network = NeuralNetwork(2, 2, 1, hidden_layer_weights=[-3.0, 1.0, 6.0, -2.0], hidden_layer_bias=0, output_layer_weights=[8.0, 4.0], output_layer_bias=0)
    # nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35,
    #                    output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)

    # f = gzip.open('mnist.pkl.gz', 'rb')
    # train_set, valid_set, test_set = cPickle.load(f)
    # f.close()
    # train_set1 = translate_data(train_set)
    # valid_set1 = translate_data(valid_set)
    # test_set1 = translate_data(test_set)

    # neuronal_network = NeuralNetwork(784, 30, 10)
    dt = DataLoader()
    train_set, valid_set, test_set = dt.get_data()

    # train_set1 = translate_data(train_set)
    # train_set = (numpy.array([[0.05, 0.1]], dtype="float32"), numpy.array([[0.01, 0.99]], dtype="float32"))
    # valid_set = (numpy.array([[0.05, 0.1]], dtype="float32"), numpy.array([[0.01, 0.99]], dtype="float32"))

    # for i in range(10):
    #     print "iteration" + str(i)
    #     size = 1000
    #     neuronal_network.train(train_set[:size])
    #     print neuronal_network.get_training_error(train_set[:size])
    #     print neuronal_network.get_test_error(test_set[:size])

    n = Network(784, 36, 10)
    n.train(train_set, 10, 0.5, 10, 0.5, 0.1, valid_set, True)
    print "Accuracy on test data : {}/{}".format(n.accuracy(test_set), len(test_set))
