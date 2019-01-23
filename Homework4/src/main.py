import keras
import numpy as np
from keras import regularizers, Input, Model, Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D, Add, K, AveragePooling2D, \
    GlobalAveragePooling2D, Lambda
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import np_utils
import matplotlib.pyplot as plt

# set GPU memory
from sklearn.decomposition import PCA
from tensorflow import int32

if 'tensorflow' == K.backend():
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def simple_scheduler(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001


def cosine_scheduler(epoch):
    rest = epoch % 20
    cat = epoch / 20
    return 0.9 * (1 / (cat + 1)) * np.cos((rest / 20) * (np.pi / 2))


def get_residual_block(input_layer, output_filters, increase=False):
    stride = (1, 1)
    if increase:
        stride = (2, 2)

    output1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(input_layer))
    convolution_1 = Conv2D(output_filters, kernel_size=(3, 3), strides=stride, padding='same',
                           kernel_initializer="he_normal",
                           kernel_regularizer=regularizers.l2(1e-4))(output1)
    output2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(convolution_1))
    convolution_2 = Conv2D(output_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                           kernel_initializer="he_normal",
                           kernel_regularizer=regularizers.l2(1e-4))(output2)

    if increase:
        projection = Conv2D(output_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(1e-4))(output1)
        block = Add()([convolution_2, projection])
    else:
        block = Add()([convolution_2, input_layer])

    return block


def build_residual_model():
    stack_n = 5

    network_input = Input(shape=(32, 32, 3))
    nn = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(network_input)

    # input: 32x32x16 output: 32x32x16
    for i in range(stack_n):
        nn = get_residual_block(nn, 16, False)

    # input: 32x32x16 output: 16x16x32
    nn = get_residual_block(nn, 32, True)
    for i in range(1, stack_n):
        nn = get_residual_block(nn, 32, False)

    # input: 16x16x32 output: 8x8x64
    nn = get_residual_block(nn, 64, True)
    for i in range(1, stack_n):
        nn = get_residual_block(nn, 64, False)

    nn = BatchNormalization(momentum=0.9, epsilon=1e-5)(nn)
    nn = Activation('relu')(nn)
    nn = GlobalAveragePooling2D()(nn)

    # input: 64 output: 10
    nn = Dense(10, activation='softmax', kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(1e-4))(nn)

    model = Model(network_input, nn)
    model.summary()
    return model


def add_basic_block(model, filter_size, kernel_size, weight_decay=1e-4):
    model.add(Conv2D(filter_size, (kernel_size, kernel_size),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())


def add_pooling_dropout(model, pooling_size, dropout):
    model.add(MaxPooling2D(pool_size=(pooling_size, pooling_size)))
    model.add(Dropout(dropout))


def expand_conv(init, base, k, stride):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    shortcut = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init)
    shortcut = Activation('relu')(shortcut)

    x = ZeroPadding2D((1, 1))(shortcut)
    x = Convolution2D(base * k, (3, 3), strides=stride, padding='valid', kernel_initializer='he_normal')(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(base * k, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='he_normal')(x)

    # Add shortcut

    shortcut = Convolution2D(base * k, (1, 1), strides=stride, padding='same', kernel_initializer='he_normal')(shortcut)

    m = Add()([x, shortcut])

    return m


def conv_block(input, n, k=1, dropout=0.3):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(n * k, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(n * k, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)

    m = Add()([init, x])
    return m


def normalize_data(x):
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    return (x - mean) / std


def random_cropping(x):
    dx = 24
    dy = 24

    start_x = np.random.randint(0, 32 - 24 + 1)
    start_y = np.random.randint(0, 32 - 24 + 1)
    return x[start_y:(start_y + dy), start_x:(start_x + dx), :]


def get_model():
    reg = l2(1e-4)  # L2 or "ridge" regularisation
    # reg = None
    num_filters = 128
    ac = 'elu'
    drop_dense = 0.5
    drop_conv = 0

    model = Sequential()
    model.add(Lambda(normalize_data, input_shape=x_train.shape[1:]))
    model.add(Lambda(random_cropping, output_shape=(24, 24, 3)))
    model.add(Conv2D(64, (3, 3), activation=ac, kernel_regularizer=reg, padding='same',
                     kernel_initializer='he_normal', use_bias=False))

    for i in range(2):
        model.add(Conv2D(num_filters, (3, 3), kernel_regularizer=reg, padding='same', kernel_initializer='he_normal',
                         use_bias=False))
        model.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 16x16x3xnum_filters
        model.add(Dropout(drop_conv))

    for i in range(2):
        model.add(Conv2D(num_filters, (3, 3), kernel_regularizer=reg, padding='same',
                         kernel_initializer='he_normal', use_bias=False))
        model.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(drop_conv))

    model.add(Conv2D(num_filters, (1, 1), kernel_regularizer=reg, padding='same',
                     kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(drop_conv))

    model.add(Conv2D(num_filters, (1, 1), kernel_regularizer=reg, padding='same',
                     kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_conv))

    model.add(Conv2D(num_filters, (3, 3), kernel_regularizer=reg, padding='same',
                     kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_conv))

    model.add(Flatten())
    # model.add(Dense(128, kernel_regularizer=reg))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(drop_dense))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    num_classes = 10
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # training
    batch_size = 128
    model = get_model()

    checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True, monitor='val_acc')
    tensorboard = TensorBoard(log_dir='./logs/simple-1x1-rmsprop-no-dropout', histogram_freq=0)

    opt_rms = keras.optimizers.rmsprop(lr=0.002, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=200,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[checkpointer, tensorboard])
