import argparse
import sys

import keras
import numpy as np
from IPython.display import clear_output
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dropout, Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from mnist import MNIST
from sklearn.utils import class_weight


def load_full_training_data(directory):
    emnist_data = MNIST(path=directory, return_type='numpy')
    return emnist_data.load_training()


codes = np.eye(47)


def transform_labels(y):
    return codes[np.array(y, dtype="int")]


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()

plot_losses = PlotLosses()


def preprocess(tensor_input):
    return tensor_input / 255.0

def schedule(epoch):
    rest = epoch % 20
    cat = epoch / 20
    return 0.9 * (1 / (cat + 1)) * np.cos((rest / 20) * (np.pi / 2))


def get_model():
    print("Build nn..")

    model = Sequential()
    # model.add(Reshape((784,), input_shape=(28, 28)))
    # model.add(Lambda(preprocess, input_shape=(784, ), output_shape=(784,)))
    model.add(Dense(500, activation='relu', input_shape=(784, )))
    model.add(Dropout(0.4))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(94, activation='relu'))
    model.add(Dense(47, activation='softmax'))

    print(model.summary())
    optimizer2 = SGD()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer2,
                  metrics=['accuracy'])
    print("Done building nn..")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train dense neural network to recognize handwritten digits and letters.')
    parser.add_argument('-td', '--train-data', help='folder with full train data')

    parse_result = parser.parse_args(sys.argv[1:])

    print("Loading training data..")

    x_train, y_train = load_full_training_data(parse_result.train_data)
    x_train = x_train.astype("float32") / 255.0
    print("Done loading training and test data..")

    print("Preprocessing data..")

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    y_train = transform_labels(y_train)

    print("Done preprocessing data..")

    print("Train..")

    model = get_model()
    checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(patience=5, verbose=1)
    lr_scheduler = LearningRateScheduler(schedule, verbose=1)
    model.fit(x_train,
              y_train,
              batch_size=128,
              epochs=200,
              verbose=1,
              validation_split=0.1,
              shuffle=True,
              class_weight=class_weights,
              callbacks=[checkpointer, plot_losses, lr_scheduler])

    print("Done training..")
