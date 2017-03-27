def center_crop_image(path, new_width, new_height):
    from PIL import Image

    im = Image.open(path).convert('RGB')
    width, height = im.size  # Get dimensions

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return im.crop((left, top, right, bottom))


def ensure_dir(file_path):
    import os
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_image(filepath, row):
    import urllib
    from PIL import Image
    urlpath = "http://maps.google.com/maps/api/staticmap?center="
    urlpath += row['Y_Coordinate'].replace(',', '.') + "," + row['X_Coordinate'].replace(',', '.')
    urlpath += "&zoom=18&size=442x442&maptype=satellite"
    urlpath += "&key=AIzaSyC9d7-JkZseVB_YW9bdIAaFCbQRLTKGaNY"
    urllib.urlretrieve(urlpath, filepath)
    image = center_crop_image(filepath, 400, 400)
    image = image.resize((100, 100), Image.ANTIALIAS)
    image.save(filepath)


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
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return (X_train, Y_train), (X_test, Y_test)


def load_csv(filename):
    X_meta = []
    Y_train = []
    X_images = []
    Coordinates = []

    import csv
    csvfile = open(filename)
    reader = csv.DictReader(csvfile, delimiter='\t')

    count = 0
    for row in reader:
        if count > 5000:
            break
        count += 1

        filepath = "nwt-data/images/" + row['ZipCode'].zfill(5) + "/" + row['House_ID'] + ".jpg"
        ensure_dir(filepath)

        # Fill image input array
        from scipy import misc
        try:
            img = misc.imread(filepath)
        except IOError:
            # Download image from GoogleMaps API
            download_image(filepath, row)
            print count
            img = misc.imread(filepath)
        X_images.extend([img])

        # Fill coordinates array
        coordinate = [
            float(row['X_Coordinate'].replace(',', '.')),
            float(row['Y_Coordinate'].replace(',', '.')),
        ]
        Coordinates.extend([coordinate])

        # Fill metadata input array
        x = [
            # float(row['X_Coordinate'].replace(',', '.')),
            # float(row['Y_Coordinate'].replace(',', '.')),
            float(hash(row['District'])) % 100,
            float(hash(row['Street'])) % 100,
            # float(row['ZipCode']),
        ]
        X_meta.extend([x])

        # Fill output array
        y = [
            int(row['ZipCode']),
        ]
        Y_train.extend([y])

    import numpy

    Coordinates = numpy.asarray(Coordinates, dtype=numpy.float32)
    X_meta = numpy.asarray(X_meta, dtype=numpy.float32)
    X_images = numpy.asarray(X_images, dtype=numpy.float32)
    Y_train = numpy.asarray(Y_train, dtype=numpy.float32)

    print ("X_images.shape", X_images.shape)
    print ("Y_train.shape", Y_train.shape)

    return Coordinates, (X_meta, X_images, Y_train)


def create_net(X_meta, X_images, Y_train):
    from keras.models import Sequential
    from keras.layers import (Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, Merge, ZeroPadding2D)

    # First define the image model
    image_processor = Sequential()

    image_processor.add(ZeroPadding2D((1, 1), input_shape=X_images.shape[1:]))
    image_processor.add(Convolution2D(64, (3, 3), activation='relu'))
    image_processor.add(ZeroPadding2D((1, 1)))
    image_processor.add(Convolution2D(64, (3, 3), activation='relu'))
    image_processor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    image_processor.add(ZeroPadding2D((1, 1)))
    image_processor.add(Convolution2D(64, (3, 3), activation='relu'))
    image_processor.add(ZeroPadding2D((1, 1)))
    image_processor.add(Convolution2D(64, (3, 3), activation='relu'))
    image_processor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    image_processor.add(ZeroPadding2D((1, 1)))
    image_processor.add(Convolution2D(64, (3, 3), activation='relu'))
    image_processor.add(ZeroPadding2D((1, 1)))
    image_processor.add(Convolution2D(64, (3, 3), activation='relu'))
    image_processor.add(ZeroPadding2D((1, 1)))
    image_processor.add(Convolution2D(64, (3, 3), activation='relu'))

    image_processor.add(Flatten())  # transform image to vector
    image_output = 1024
    image_processor.add(Dense(image_output, activation='relu'))

    # Now we create the metadata model
    metadata_processor = Sequential()
    metadata_processor.add(Dense(256, activation='relu', input_shape=X_meta.shape[1:]))
    metadata_output = 256
    metadata_processor.add(Dense(metadata_output, activation='relu'))
    # metadata_processor.add(Dropout(0.1))

    # Now we concatenate the two features and add a few more layers on top
    model = Sequential()
    # TODO: Use keras.layers.merge.concatenate layer
    model.add(Merge([image_processor, metadata_processor], mode='concat'))  # Merge is your sensor fusion buddy
    model.add(Dense(256, activation='relu', input_dim=image_output + metadata_output))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(Y_train.shape[1]))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


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


def show_prediction():
    prediction = model.predict([X_images, X_meta]).flatten()
    import matplotlib.pyplot as plt
    plt.scatter(Coordinates[:, 0], Coordinates[:, 1], c=prediction, s=0.5, cmap='jet')
    plt.show()


def save_prediction(filename):
    import cPickle as pickle
    prediction = model.predict([X_images, X_meta]).flatten()

    result = [Coordinates[:, 0], Coordinates[:, 1], Y_train, prediction]

    pickle.dump(result, open(filename, "wb"))


def show_prediction_from_file(filename):
    import cPickle as pickle
    x, y, t, prediction = pickle.load(open(filename, "rb"))

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c=prediction, s=0.5, cmap='jet')
    plt.show()


def save_model(model, filename):
    # TODO: save model of net in file
    return


def load_model(filename):
    # TODO: load model of net from file
    return


def save_weigths(model, filename):
    # TODO: save weights of net in file
    return


def load_weights(filename):
    # TODO: load weights of net from file
    return


# TODO: load all images from google maps api


def main():
    # TODO: use main function (refactor code)

    (Coordinates), (X_meta, X_images, Y_train) = load_csv("nwt-data/Gebaeude_Dresden_shuffle.csv")

    model = create_net(X_meta, X_images, Y_train)

    ## Some model and data processing constants
    batch_size = 128
    nb_epoch = 5

    history = model.fit(x=[X_images, X_meta], y=Y_train,
                        batch_size=batch_size, epochs=nb_epoch,
                        verbose=2, shuffle=True,
                        validation_split=0.1)

    show_acc()
    show_loss()
    show_prediction()
    # save_prediction("result/prediction.pkl")
    # show_prediction_from_file("result/prediction.pkl")

    return


if __name__ == "__main__":
    main()
