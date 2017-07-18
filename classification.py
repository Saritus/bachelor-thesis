from functions import ensure_dir
from csvhelper import load_csv
from visualize import show_acc, show_loss, show_prediction
from keras.callbacks import ModelCheckpoint


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
    from keras.layers import (Flatten, Dense, Convolution2D, MaxPooling2D, Merge, ZeroPadding2D, Dropout)

    # First define the image model
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=X_train[0].shape[1:]))
    model.add(Convolution2D(16, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(Y_train.shape[1], activation='softmax'))

    import keras.optimizers
    SGD = keras.optimizers.SGD(lr=0.0005, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['categorical_accuracy'])

    return model


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

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    # load weights from file
    if file_weights:
        model.load_weights(file_weights)

    return model


def main():
    (Coordinates), (X_train, Y_train) = load_csv("nwt-data/Schulen Sachsen/schulen_sachsen_shuffle.csv")

    model = create_net(X_train, Y_train)
    # model = load_model("models/first_try.json", "models/first_try.h5")

    ## Some model and data processing constants
    batch_size = 64
    nb_epoch = 15

    checkpointer = ModelCheckpoint(filepath='models/schoolsSA-epoch{epoch:02d}-acc{val_categorical_accuracy:.3f}.hdf5',
                                   verbose=1, monitor='val_categorical_accuracy',
                                   save_best_only=True)
    # history = model.fit(x=X_train, y=Y_train,
    #                    batch_size=batch_size, epochs=nb_epoch,
    #                    verbose=1, shuffle=True,
    #                    validation_split=0.1, callbacks=[checkpointer])

    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )

    train_generator = train_datagen.flow_from_directory(
        'images/google-address/SchoolsSA_train',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        # save_to_dir='images/google-address/SchoolsSA_train_flow'
    )

    validation_generator = test_datagen.flow_from_directory(
        'images/google-address/SchoolsSA_val',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        # save_to_dir='images/google-address/SchoolsSA_val_flow'
    )

    # fine-tune the model
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=1024,
        epochs=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=128,
        callbacks=[checkpointer])

    prediction = model.predict(X_train)
    print Y_train[:10]
    print prediction[:10]

    # save weights of net
    # save_model(model, "models/first_try.json", "models/first_try.h5")

    show_acc(history)
    show_loss(history)
    # show_prediction(model, X_train, Coordinates)
    # save_prediction(model, X_train, Coordinates, "result/prediction_test.pkl")
    # show_prediction_from_file("result/prediction_test.pkl")

    return


if __name__ == "__main__":
    main()
