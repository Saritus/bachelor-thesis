import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.updates import adam

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import ReshapeLayer
from lasagne.nonlinearities import softmax

from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_conv_weights

import sklearn.metrics

import sys
import os
import gzip
import pickle
import numpy
import math
import time
import matplotlib.pyplot as plt

from pylab import imshow, show, cm, imsave
from PIL import Image

from urllib import urlretrieve


def pickle_load(f):
    return pickle.load(f)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def show_image(x, width=11, height=11):
    #nx = numpy.array(x, dtype=numpy.uint8).reshape(width, height, 3)
    nx = numpy.array(x).reshape(width, height, 3)
    image = Image.fromarray(nx)
    imshow(image, interpolation='nearest')
    show()


from numpngw import write_png
def save_png(array, filename):
    write_png(filename, numpy.swapaxes(array, 0, 1))


def label_to_color(array, color_table):
    imagearray = []
    for i in range(len(array)):
        imagearray.append(color_table[array[i]])
    imagearray = numpy.array(imagearray, dtype='uint8')
    return imagearray

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
#DATA_FILENAME = "satellite1.png_50_50_4_4_5.pkl"
PATCHSIZE = 11

savedir = "split_valid_test_1"
if not os.path.exists(savedir):
    os.makedirs(savedir)


def split_rgb(x):
    rgb = []
    for elem in x:
        pic = []
        pic.extend([elem[0::3].reshape(PATCHSIZE, PATCHSIZE)])
        pic.extend([elem[1::3].reshape(PATCHSIZE, PATCHSIZE)])
        pic.extend([elem[2::3].reshape(PATCHSIZE, PATCHSIZE)])
        rgb.extend([pic])
    rgb = numpy.array(rgb, dtype=numpy.int8)
    return rgb


def load_data(file_train, file_test):
    """Get data with labels, split into training, validation and test set."""
    with open(file_test, 'rb') as f:
        data = pickle_load(f)

    x_test, y_test = data[0]

    x_test_rgb = split_rgb(x_test)

    x_test = x_test.reshape((-1, 1, PATCHSIZE, PATCHSIZE, 3))

    with open(file_train, 'rb') as f:
        data = pickle_load(f)



    x_train, y_train = data[0]

    x_train = numpy.array(x_train, dtype=numpy.float32)
    y_train = numpy.array(y_train, dtype=numpy.int8)

    x_train_rgb = split_rgb(x_train)

    x_train = x_train.reshape((-1, 1, PATCHSIZE, PATCHSIZE, 3))

    x_valid = x_train
    y_valid = y_train
    x_valid_rgb = x_train_rgb

    return dict(
        x_train=x_train,
        y_train=y_train,
        x_train_rgb=x_train_rgb,
        x_valid=x_valid,
        y_valid=y_valid,
        x_valid_rgb=x_valid_rgb,
        x_test=x_test,
        y_test=y_test,
        x_test_rgb=x_test_rgb,
        num_examples_train=x_train.shape[0],
        input_dim=x_train.shape[1],
        output_dim=len(data[1]),
        color_table=data[1]
    )

data = load_data("satellite1.png_50_50_4_4_5.pkl", "satellite2.png_50_50_4_4_5.pkl")


def create_mlp():
    net = NeuralNet(
        layers=[('input', InputLayer),
            #('reshape', FlattenLayer),
            ('hidden1', DenseLayer),
            #('dropout1',layers.DropoutLayer), 
            ('hidden2', DenseLayer),
            #('dropout2',layers.DropoutLayer), 
            #('hidden3', DenseLayer),
            ('output', DenseLayer),
        ],
                
        # layer parameters:
        input_shape=(None, 1, PATCHSIZE, PATCHSIZE, 3),
        #reshape_shape=(([0], 11*11)),outdim
        #reshape_outdim=1,
        hidden1_num_units=5000,  # number of units in 'hidden' layer
        hidden2_num_units=5000,  # number of units in 'hidden' layer
        #hidden3_num_units=2601,  # number of units in 'hidden' layer

        #dropout1_p=0.15,
        #dropout2_p=0.15,
                
        output_nonlinearity=softmax,
        output_num_units=data['output_dim'],  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=adam,
        update_learning_rate=0.001,
        #update_momentum=0.9,

        max_epochs=1,
        verbose=1,
    )
    return net


def create_cnn():
    net = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                #('maxpool1', layers.MaxPool2DLayer),
                #('conv2d2', layers.Conv2DLayer),
                #('maxpool2', layers.MaxPool2DLayer),
                #('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                #('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 3, PATCHSIZE, PATCHSIZE),
        # layer conv2d1
        conv2d1_num_filters=1024,
        conv2d1_filter_size=5,
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # layer maxpool1
        #maxpool1_pool_size=(2, 2),
        # layer conv2d2
        #conv2d2_num_filters=512,
        #conv2d2_filter_size=(3, 3),
        #conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        #maxpool2_pool_size=(2, 2),
        # dropout17
        #dropout1_p=0.1,    
        # dense
        dense_num_units=5000,
        #dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        #dropout2_p=0.1,    
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=data['output_dim'],
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.95,
        max_epochs=1,
        verbose=1
    )
    return net


def create_ae():
    conv_filters = 8
    deconv_filters = 1
    filter_sizes = 1
    ae = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            #('conv', layers.Conv2DLayer),
            #('pool', layers.MaxPool2DLayer),
            #('flatten', ReshapeLayer),  # output_dense
            #('encode_layer', layers.DenseLayer),
            ('hidden', layers.DenseLayer),  # output_dense
            #('unflatten', ReshapeLayer),
            #('unpool', Unpool2DLayer),
            #('deconv', layers.Conv2DLayer),
            ('output_layer', DenseLayer),
        ],
        input_shape=(None, 1, PATCHSIZE, PATCHSIZE),

        #conv_num_filters=conv_filters,
        #conv_filter_size = (filter_sizes, filter_sizes),
        #conv_nonlinearity=None,

        #pool_pool_size=(2, 2),

        #flatten_shape=([0], -1), # not sure if necessary?

        #encode_layer_num_units = 121,

        #hidden_num_units = deconv_filters * (PATCHSIZE + filter_sizes - 1) ** 2 / 4,
        hidden_num_units = 121,

        #unflatten_shape=([0], deconv_filters, 11, 11 ),

        #unpool_ds=(2, 2),

        #deconv_num_filters = deconv_filters,
        #deconv_filter_size = (filter_sizes, filter_sizes),
        #deconv_nonlinearity=None,

        #output_layer_shape =([0], 121),
        output_layer_num_units = 121,

        update_learning_rate = 0.01,
        update_momentum = 0.975,
        #batch_iterator_train=FlipBatchIterator(batch_size=128),
        regression=True,
        max_epochs= 1,
        verbose=1,
    )
    return ae

net1 = create_cnn()


def predict(pdata, width, height, filename):
    prediction = net1.predict(pdata)

    imagearray = label_to_color(prediction, data['color_table'])

    imagearray = imagearray.reshape(
        width,  # first image dimension (vertical)
        height,  # second image dimension (horizontal)
        3 # rgb
    )
    image = Image.fromarray(imagearray).transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
    image.save(filename, 'png')
    return prediction

def save_denselayer(layername, filetype="tif", width=110, height=110):
    layer0_values = layers.get_all_param_values(net1.layers_[layername])
    for neuro in range(0, layer0_values[0].shape[1]):
        layer0_1 = [layer0_values[0][i][neuro] for i in range(len(layer0_values[0]))]
        if filetype != "tif":
            layer0_1 = [i * 256 for i in layer0_1]
        layer0_1 = numpy.asarray(layer0_1)
        layer0_1 = layer0_1.reshape(
            11,  # first image dimension (vertical)
            11  # second image dimension (horizontal)
        )
        image = Image.fromarray(layer0_1)
        if filetype == "tif":
            image.resize((width, height), Image.NEAREST).save(savedir + "/" + layername + str(neuro).zfill(4) + '.tif', "TIFF")
        elif filetype == "png":
            image.convert('RGB').resize((width, height), Image.NEAREST).save(savedir + "/" + str(neuro).zfill(4) + '.png', "PNG")
        else:
            sys.stderr.write('Filetype is not supported')


def show_conv2dlayer(layername, filename=None):
    plot_conv_weights(net1.layers_[layername], figsize=(11, 11))
    if filename:
        plt.savefig(savedir+'/'+filename)
        #plt.clf()
    plt.show()


def save_score(filename):
    # Save score to file
    with open(savedir + '/' + filename, 'a') as filept:
        filept.write(str(net1.score(data['x_train'], data['y_train'])) + "\n")


def confusion_matrix(xtest, ytest):
    preds = net1.predict(data['x_test'])
    colorm = sklearn.metrics.confusion_matrix(data['y_test'], preds)
    plt.matshow(colorm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def show_output(layername):
    instance = data['x_train'][260].reshape(1, 1, 11, 11)
    pred = layers.get_output(net1.layers_[layername], instance, deterministic=False)
    N = pred.shape[1]
    #print pred[0][0].eval()
    print N
    #plt.bar(range(N), pred.ravel())
    plt.show()


def restrict_layer(layername, minimum=None, maximum=None):
    layer_values = layers.get_all_param_values(net1.layers_[layername])
    for to_neuron in range(0, layer_values[0].shape[0]):
        if minimum:
            layer_values[0][to_neuron][layer_values[0][to_neuron] < minimum] = minimum
        if maximum:
            layer_values[0][to_neuron][layer_values[0][to_neuron] > maximum] = maximum

    layers.set_all_param_values(net1.layers_[layername], layer_values)


def write_to_file(filename, string):
    f = open(filename, 'a')
    f.write(string + '\n') # python will convert \n to os.linesep
    f.close() # you can omit in most cases as the destructor will call it


def array_to_string(array):
    string = ""
    for elem in array:
        string += str(elem)+'\t'
    return string


def shuffle_data():
    # Randomize new order
    length = len(data['x_train'])
    new_order = range(0, length)
    numpy.random.shuffle(new_order)

    # Shuffle x_train
    data_x = [data['x_train'][i] for i in new_order]
    data_x = numpy.asarray(data_x)
    data['x_train'] = data_x

    # Shuffle y_train
    data_y = [data['y_train'][i] for i in new_order]
    data_y = numpy.asarray(data_y)
    data['y_train'] = data_y


def calculate_prediction_result(xdata, ydata):
    prediction = net1.predict(xdata)

    right = numpy.zeros(data['output_dim'])
    overall = numpy.zeros(data['output_dim'])
    for i in range(len(ydata)):
        if ydata[i]==prediction[i]:
            right[[ydata[i]]]+=1
        overall[[ydata[i]]]+=1

    relative = numpy.zeros(len(overall))
    for i in range(len(relative)):
        relative[i]=right[i] / overall[i]
    return relative

def main():

    print("Got %i testing datasets." % len(data['x_train']))

    #for img in range(0, 10):
    #    show_image(data['x_train'][img])

    zero=0
    one=0
    two=0
    for y in data['y_train']:
        if y==0:
            zero+=1
        elif y==1:
            one+=1
        elif y==2:
            two+=1
        else:
            print 'Error'
    print zero, one, two

    #imshow(data['y_train'].reshape(148, 143))
    #show()

    imshow(data['x_train_rgb'][0][0], interpolation='nearest', cmap='gray')
    #show()
    imshow(data['x_train_rgb'][0][1], interpolation='nearest', cmap='gray')
    #show()
    imshow(data['x_train_rgb'][0][1], interpolation='nearest', cmap='gray')
    #show()

    #shuffle_data()
    for x in range(1, 1000):
        # Train the network
        net1.fit(data['x_train_rgb'], data['y_train'])
        validpred = predict(data['x_valid_rgb'], 148, 143, savedir+'/valid'+str(x).zfill(3)+'.png')
        testpred = predict(data['x_test_rgb'], 148, 143, savedir+'/test'+str(x).zfill(3)+'.png')
        #write_to_file(savedir + "/valid.txt", array_to_string(calculate_prediction_result(data['x_valid_rgb'], data['y_valid'])))
        #write_to_file(savedir + "/test.txt", array_to_string(calculate_prediction_result(data['x_test_rgb'], data['y_test'])))

        #imshow(predict(308, 296, savedir+'/'+str(x).zfill(2)+'.png'))
        #show()
        #restrict_conv_layer()
        #restrict_layer("encode_layer", 0.01)
        #restrict_layer(2, 0)
        #if x%10==0:
        #    confusion_matrix()


        #show_conv2dlayer('conv2d1', '9x9')

    #save_denselayer('encode_layer')
    #save_denselayer('hidden')

    '''
    for i in range(1, 12770):
        result = net1.predict([data['x_test'][i]])
        #print str(i) + " - " + str(result.max())
        if result.max() > 0.2:
            print i
            #print result.shape
            print result
            imagearray = numpy.array([value * 256 for value in result])
            show_image(data['x_test'][i])
            show_image(imagearray)
            #show_image(data['x_ae'][i])
    '''

    '''
    #imshow(image, cmap=cm.gray)
    #show()
    #image.save("epochs" + str(net1.max_epochs).zfill(4) + "_hidden_" + str(net1.hidden1_num_units).zfill(4) +'_'+str(net1.hidden2_num_units).zfill(4)+'_'+str(net1.hidden3_num_units).zfill(4)+ '.png',"PNG")

    image.save(savedir + "/epoch" +str(x * net1.max_epochs).zfill(4) + '.png',"PNG")
    #print savedir + "/epoch" +str(x * net1.max_epochs).zfill(4) + '.png saved'
    '''

    # print net1
    # print net1.score(data['x_train'], data['y_train'])
    # numpy.set_printoptions(threshold=numpy.inf)


def restrict_conv_layer():
    layer_values = layers.get_all_param_values(net1.layers_['conv2d1'])

    for to_neuron in range(0, layer_values[0].shape[0]):
        #layer_values[0][to_neuron][ layer_values[0][to_neuron] < -1] = -1
        layer_values[0][to_neuron][layer_values[0][to_neuron] > 0] = 0

    layers.set_all_param_values(net1.layers_['conv2d1'], layer_values)


if __name__ == '__main__':
    main()
