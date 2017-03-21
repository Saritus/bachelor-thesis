def show_prediction_from_file(filename):
    import cPickle as pickle
    x, y, t, prediction = pickle.load(open(filename, "rb"))

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c=t, s=0.5, cmap='jet')
    plt.show()
    plt.scatter(x, y, c=prediction, s=0.5, cmap='jet')
    plt.show()
    
show_prediction_from_file("prediction.pkl")
