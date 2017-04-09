def show_acc(history):
    # ACC VS VAL_ACC
    import matplotlib.pyplot as plt
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy ACC VS VAL_ACC')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def show_loss(history):
    # LOSS VS VAL_LOSS
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss LOSS vs VAL_LOSS')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def show_prediction(model, X_train, Coordinates):
    prediction = model.predict(X_train).flatten()
    import matplotlib.pyplot as plt
    plt.scatter(Coordinates[:, 0], Coordinates[:, 1], c=prediction, s=0.5, cmap='jet')
    plt.show()


def save_prediction(model, X_train, Coordinates, filename):
    import cPickle as pickle
    prediction = model.predict(X_train).flatten()

    result = [Coordinates[:, 0], Coordinates[:, 1], prediction]

    pickle.dump(result, open(filename, "wb"))


def show_prediction_from_file(filename):
    import cPickle as pickle
    x, y, prediction = pickle.load(open(filename, "rb"))

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c=prediction, s=0.5, cmap='jet')
    plt.show()


def point_plot(x_array, y_array, color_array):
    import matplotlib.pyplot as plt
    plt.scatter(x_array, y_array, c=color_array, s=0.5, cmap='jet')
    plt.show()


def main():
    return


if __name__ == "__main__":
    main()
