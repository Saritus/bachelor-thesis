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
    point_plot(Coordinates[:, 0], Coordinates[:, 1], prediction)


def save_prediction(model, X_train, Coordinates, filename):
    import cPickle as pickle
    prediction = model.predict(X_train).flatten()

    result = [Coordinates[:, 0], Coordinates[:, 1], prediction]

    pickle.dump(result, open(filename, "wb"))


def show_prediction_from_file(filename):
    import cPickle as pickle
    x, y, prediction = pickle.load(open(filename, "rb"))
    point_plot(x, y, prediction)


def point_plot(x_array, y_array, color_array=None):
    import matplotlib.pyplot as plt
    if color_array:
        plt.scatter(x_array, y_array, c=color_array, s=0.5, cmap='jet')
    else:
        plt.scatter(x_array, y_array, s=0.5)
    plt.show()


def main():
    return


if __name__ == "__main__":
    main()
