from keras import Input, Model
from keras.engine.saving import load_model

from mnist import MNIST
import numpy as np

def load_full_test_data(directory):
    emnist_data = MNIST(path=directory, return_type='numpy')
    return emnist_data.load_testing()


codes = np.eye(47)


def transform_labels(y):
    return codes[np.array(y, dtype="int")]


if __name__ == "__main__":
    x_test, y_test = load_full_test_data("test")
    model = load_model('model.h5')

    for l in model.layers:
        print(l.get_config())
    # x_test = x_test.reshape((x_test.shape[0], 28, 28))
    x_test = x_test.astype("float32") / 255

    y_test = transform_labels(y_test)
    print(model.evaluate(x_test, y_test))
