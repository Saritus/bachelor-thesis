def load_mnist():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    from keras.utils import np_utils

    # input image dimensions
    img_rows, img_cols = 28, 28
    nb_classes = 10

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # X_train = X_train[:5000]
    # X_test = X_test[:5000]
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    # Y_train = Y_train[:5000]
    # Y_test = Y_test[:5000]

    return (X_train, Y_train), (X_test, Y_test)


def load_csv(filename):
    X_train = []
    Y_train = []

    import csv
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')

        for row in reader:
            x = [
                float(row['X_Coordinate'].replace(',', '.')),
                float(row['Y_Coordinate'].replace(',', '.')),
                float(hash(row['District'])) % 100,
                float(hash(row['Street'])) % 100
            ]
            X_train.extend([x])
            y = [
                int(row['ZipCode']) % 10
                # int(row['Flag_Coordinates'])
            ]
            Y_train.extend([y])

    import numpy
    X_train = numpy.asarray(X_train, dtype=numpy.float32)
    Y_train = numpy.asarray(Y_train, dtype=numpy.float32)
    X_test = numpy.asarray(X_train, dtype=numpy.float32)
    Y_test = numpy.asarray(Y_train, dtype=numpy.float32)

    return (X_train, Y_train), (X_test, Y_test)


(X_train, Y_train), (X_test, Y_test) = load_csv("nwt-data/Gebaeude_Dresden.csv")


def create_net():
    from keras.models import Sequential
    from keras.layers import (Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D)

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    model = Sequential()

    print ("X_train", X_train.shape)
    print ("Y_train", Y_train.shape)

    model.add(Dense(512, input_shape=(X_train.shape[1:])))
    # model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), padding='valid', input_shape=(X_train.shape[1:])))
    model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))

    # model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(Y_train.shape[1]))
    # model.add(Activation('softmax'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = create_net()

## Some model and data processing constants
batch_size = 128
nb_epoch = 12

history = model.fit(x=[X_train], y=[Y_train],
                    batch_size=batch_size,
                    epochs=nb_epoch, verbose=2,
                    shuffle=True,
                    validation_split=0.1
                    # validation_data=(X_test, Y_test)
                    )


def show_acc():
    # ACC VS VAL_ACC
    import matplotlib.pyplot as plt
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy ACC VS VAL_ACC')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


show_acc()


def show_loss():
    # LOSS VS VAL_LOSS
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss LOSS vs VAL_LOSS')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


show_loss()
