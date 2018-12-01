import argparse
import sys

import keras

import os

import numpy as np
from IPython.display import clear_output
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, Dense, regularizers
from keras.optimizers import SGD, Adam
import process_data
from matplotlib import pyplot as plt

import seaborn as sns

PREPROCESSED_DATA = "processed_data"

TRAIN_DATA_X = os.path.join(PREPROCESSED_DATA, "x_full_train.npy")
TRAIN_DATA_Y = os.path.join(PREPROCESSED_DATA, "y_full_train.npy")

TEST_DATA_X = os.path.join(PREPROCESSED_DATA, "x_full_test.npy")
TEST_DATA_Y = os.path.join(PREPROCESSED_DATA, "y_full_test.npy")


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


def plot_distribution(data):
    sns.countplot(data)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train dense neural network to recognize handwritten digits and letters.')
    parser.add_argument('-d', '--data', help='folder with full data')
    parser.add_argument('-npy', '--numpy-data', help='train using preprocessed data saved in numpy arrays')

    parse_result = parser.parse_args(sys.argv[1:])

    print("Loading training data..")

    if parse_result.data is not None:
        x_train, y_train = process_data.load_full_training_data(parse_result.data)
        x_test, y_test = process_data.load_full_test_data(parse_result.data)

        print("Done loading training data..")

        print("Preprocessing data..")
        x_train = np.apply_along_axis(process_data.rotate, 1, x_train.astype("float32")) / 255
        x_train = np.apply_along_axis(process_data.zoom, 1, x_train.astype("float32"))


    else:
        x_train, y_train = process_data.load_full_numpy_arrays("train", parse_result.numpy_data)
        x_test, y_test = process_data.load_full_numpy_arrays("test", parse_result.numpy_data)
        print("Done loading training data..")
        print("Preprocessing data..")

    y_test = process_data.transform_labels(y_test)
    y_train = process_data.transform_labels(y_train)

    print("Done preprocessing data..")

    print("Build nn..")
    input_layer = Input(shape=(784,))
    dropout_1 = Dropout(0.6)(input_layer)
    dense_2 = Dense(1024, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.6)(dense_2)
    output_layer = Dense(47, activation='softmax')(dropout_2)
    model = Model(inputs=input_layer, outputs=output_layer)

    print(model.summary())
    model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
                  optimizer="adam",
                  metrics=['accuracy'])  # reporting the accuracy
    checkpointer = ModelCheckpoint('model-emnist-nn.h5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(patience=3, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.001)
    print("Done building nn..")

    print("Train..")
    history = model.fit(x_train, y_train,  # Train the model using the training set...
                        batch_size=512, epochs=20,
                        verbose=1, validation_split=0.1,
                        callbacks=[earlystopper, checkpointer, reduce_lr,
                                   plot_losses])  # ...holding out 10% of the data for validation
    print("Done training..")

    print("Test..")
    print(model.evaluate(x_test, y_test))
    print("Done testing..")
