from keras.datasets import mnist 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

from keras.models import Sequential
from keras.layers import (Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D)
 
 
## Some model and data processing constants
 
batch_size = 128
nb_classes = 10
nb_epoch = 12
 
# input image dimensions
img_rows, img_cols = 28, 28
 
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
 
 
model = Sequential()

model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), border_mode='same', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
#model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
