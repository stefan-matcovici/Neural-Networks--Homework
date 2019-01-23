import keras
from IPython.core.display import clear_output
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

weight_decay = 0.0005





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

if __name__ == "__main__":
    from keras.layers import Input
    from keras.models import Model

    init = (32, 32, 3)

    wrn_28_10 = create_wide_residual_network(init, nb_classes=10, N=2, k=2, dropout=0.25)

    wrn_28_10.summary()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    # training
    batch_size = 128

    checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

    opt_rms = SGD(lr=0.1, momentum=0.9, decay=0.0005, nesterov=True)
    wrn_28_10.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    # wrn_28_10.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    #                     steps_per_epoch=x_train.shape[0] // batch_size,
    #                     epochs=125,
    #                     verbose=1,
    #                     callbacks=[checkpointer])

    wrn_28_10.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=125,
                  callbacks=[checkpointer, plot_losses],
                  validation_split=0.1,
                  verbose=1)
