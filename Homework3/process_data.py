import os

import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from mnist import MNIST
from matplotlib import pyplot as plt
import cv2

codes = np.eye(47)


def save_trimmed_numpy_arrays(size, train_x, train_y, s):
    trimmed_train_x = train_x[:size, :]
    trimmed_train_y = train_y[:size]

    np.save("x_trim_" + s, trimmed_train_x)
    np.save("y_trim_" + s, trimmed_train_y)


def save_full_numpy_arrays(train_x, train_y, s):
    np.save("x_full_" + s, train_x)
    np.save("y_full_" + s, train_y)


def transform_labels(y):
    return codes[np.array(y, dtype="int")]


def load_trimmed_numpy_arrays(s):
    return np.load("x_trim_" + s + ".npy"), np.load("y_trim_" + s + ".npy")


def load_full_numpy_arrays(s, directory):
    return np.load(os.path.join(directory, "x_full_" + s + ".npy")), np.load(
        os.path.join(directory, "y_full_" + s + ".npy"))


def load_full_training_data(directory):
    emnist_data = MNIST(path=directory, return_type='numpy')
    return emnist_data.load_training()


def load_full_test_data(directory):
    emnist_data = MNIST(path=directory, return_type='numpy')
    return emnist_data.load_testing()


def show_image(pixels):
    plt.imshow(pixels, cmap='gray')
    plt.show()


def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])


def zoom(image):
    image = image.reshape((28, 28))

    while np.sum(image[0]) == 0:
        image = image[1:]
    while np.sum(image[:, 0]) == 0:
        image = np.delete(image, 0, 1)
    while np.sum(image[-1]) == 0:
        image = image[:-1]
    while np.sum(image[:, -1]) == 0:
        image = np.delete(image, -1, 1)
    image = cv2.resize(image, (28, 28))

    return image.reshape([28 * 28])


def preprocess_image(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)

    while np.sum(image[0]) == 0:
        image = image[1:]
    while np.sum(image[:, 0]) == 0:
        image = np.delete(image, 0, 1)
    while np.sum(image[-1]) == 0:
        image = image[:-1]
    while np.sum(image[:, -1]) == 0:
        image = np.delete(image, -1, 1)
    image = cv2.resize(image, (28, 28))

    return image.reshape([28 * 28])


if __name__ == "__main__":
    # x_train, y_train = load_full_training_data("data")
    x_train, y_train = load_full_numpy_arrays("train_unprocessed", "processed_data")

    print("Preprocessing..")
    # x_train = np.apply_along_axis(preprocess_image, 1, x_train.astype("float32")) / 255
    print("Done")

    indexes = [[i for i, x in enumerate(y_train) if x == label] for label in range(47)]
    max_samples = max([len(x) for x in indexes])

    remaining_samples = [max_samples-len(x) for x in indexes]




