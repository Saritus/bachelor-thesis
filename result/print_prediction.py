def show_prediction_from_file(filename):
    import cPickle as pickle
    x, y, prediction = pickle.load(open(filename, "rb"))

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c=prediction, s=0.5, cmap='jet')
    
show_prediction_from_file("prediction.pkl")
