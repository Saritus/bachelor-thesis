from functions import center_crop_image, ensure_dir, download_image
from loadcsv import load_csv
from visualize import show_acc, show_loss, show_prediction, save_prediction, show_prediction_from_file


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


def create_net(X_train, Y_train):
    from keras.models import Sequential
    from keras.layers import (Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, Merge, ZeroPadding2D)

    # First define the image model
    image_processor = Sequential()

    image_processor.add(ZeroPadding2D((1, 1), input_shape=X_train[0].shape[1:]))
    image_processor.add(Convolution2D(32, (3, 3), activation='relu'))
    image_processor.add(ZeroPadding2D((1, 1)))
    image_processor.add(Convolution2D(32, (3, 3), activation='relu'))
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
    metadata_processor.add(Dense(256, activation='relu', input_shape=X_train[1].shape[1:]))
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

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    return model


# TODO: load all images from google maps api

def save_model(model, file_architecture, file_weights=None):
    # convert model to json string
    json_string = model.to_json()

    # save json string in file
    import json
    ensure_dir(file_architecture)
    with open(file_architecture, 'w') as outfile:
        json.dump(json_string, outfile)

    # save weights of model in file
    if file_weights:
        ensure_dir(file_weights)
        model.save_weights(file_weights)


def load_model(file_architecture, file_weights=None):
    # load json string from file
    import json
    with open(file_architecture) as data_file:
        json_string = json.load(data_file)

    # model reconstruction from json string
    from keras.models import model_from_json
    model = model_from_json(json_string)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    # load weights from file
    if file_weights:
        model.load_weights(file_weights)

    return model


def main():
    (Coordinates), (X_train, Y_train) = load_csv("nwt-data/Gebaeude_Dresden_shuffle.csv")

    model = create_net(X_train, Y_train)
    # model = load_model("models/first_try.json", "models/first_try.h5")

    ## Some model and data processing constants
    batch_size = 128
    nb_epoch = 50

    history = model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=nb_epoch,
                        verbose=2, shuffle=True, validation_split=0.1)

    # TODO: add a callback for fit function (link: https://keras.io/callbacks/#create-a-callback)

    # save weights of net
    save_model(model, "models/first_try.json", "models/first_try.h5")

    # show_acc(history)
    show_loss(history)
    show_prediction(model, X_train, Coordinates)
    # save_prediction(model, X_train, Coordinates, "result/prediction_test.pkl")
    # show_prediction_from_file("result/prediction_test.pkl")

    # TODO: add function for csv output (combination of input and output)

    return


if __name__ == "__main__":
    main()

    # TODO: move functions to seperate python files (next: image.py)