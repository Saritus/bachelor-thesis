# -*- coding: cp1252 -*-
# imports
from PIL import Image
import glob, os
import numpy
import sys
#import theano
import cPickle
import time

def createDir(foldername):
    # create folder if necessary
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    return foldername

# set global variables
inputImageX='satellite1.png'
#inputImageX='roadmap.png'
inputImageY='roadmap1.png'
outputDir=createDir('walls_py')
width=50
height=50
step_width=4
step_height=4
innerstep=1

from numpngw import write_png
def save_png(array, filename):
    write_png(filename, numpy.swapaxes(array, 0, 1))

from pylab import imshow, show, cm
def show_image(x, width=11, height=11):
    x = x.reshape(
        width,  # first image dimension (vertical)
        height,  # second image dimension (horizontal)
        3
    )
    image = Image.fromarray(x, 'RGB')
    imshow(image, interpolation='nearest')
    show()

# open imageX and get size
imX = Image.open(inputImageX).convert('RGB')
pixX = imX.load()
imageX=[]
for x in range(0, width):
    for y in range(0, height):
        imageX.extend(pixX[x,y])
imageX = numpy.array(imageX, dtype=numpy.uint8)
imageX = imageX.reshape(
    width,  # first image dimension (vertical)
    height,  # second image dimension (horizontal)
    3
)

# open imageY and get size
imY = Image.open(inputImageY).convert('RGB')
pixY = imY.load()
imageY=[]
for x in range(0, width):
    for y in range(0, height):
        imageY.extend(pixY[x,y])
imageY = numpy.array(imageY, dtype=numpy.uint8)
imageY = imageY.reshape(
    width,  # first image dimension (vertical)
    height,  # second image dimension (horizontal)
    3
)

save_png(imageX, 'imX.png')
save_png(imageY, 'imY.png')


im_width, im_height = imX.size
print ("width = " + str(im_width) + " _ height = " + str(im_height))

#                        Schwarz    Weiß             Gelb           Braun        Blau
#color_table = np.array([[0, 0, 0], [255, 255, 255], [255, 207, 0], [127, 0, 0], [0, 223, 255]])
color_table = []

def get_color_code(pixel):
    for index in range(len(color_table)):
        if numpy.array_equal(pixel, color_table[index]):
            return index
    color_table.append(pixel)
    sys.stderr.write("color is not in color_table\n")
    print (color_table)
    #print ((color_table)-1)
    return len(color_table)-1

def create_pkl_file(set_x, set_y, filename=inputImageX+'.pkl'):
    train_set_x = numpy.array(set_x)
    train_set_y = numpy.array(set_y)
    train_set = train_set_x, train_set_y
    data_set = [train_set, numpy.array(color_table, dtype='uint8')]
    f = open(filename, 'wb')
    cPickle.dump(data_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

print (im_width-width)/step_width + 1  # first image dimension (vertical)
print (im_height-height)/step_height + 1

# Python
#labels = np.zeros((im_width, im_height), dtype=np.uint8)
set_x = []
set_y = []
for x in range(0, im_width, step_width):
    start_time = time.time()
    if (im_width-x) < width:
        break
    for y in range(0, im_height, step_height):
        if (im_height-y) < height:
            break
        localMapX = numpy.zeros((height / innerstep + 1, width / innerstep + 1, 3), dtype=numpy.uint8)
        for indexi, i in enumerate(range(x, min(x+width+1, im_width), innerstep)):
            for indexj, j in enumerate(range(y, min(y+height+1, im_height), innerstep)):
                localMapX[indexi, indexj] = pixX[i, j]
        set_y.append(get_color_code(pixY[x,y]))
        #save_png(localMapX, outputDir + '/' + str(x).zfill(4) + '_' + str(y).zfill(4) + '.png')
        set_x.append(localMapX.flatten())
        #show_image(localMapX, width+1, height+1)
    print str(x)+' - ' + str(time.time() - start_time)
create_pkl_file(set_x, set_y, '50_50_4_4_1.pkl')

from pylab import imshow, show, cm
def label_to_color(array, color_table):
    imagearray = []
    for i in range(len(array)):
        imagearray.append(color_table[array[i]])
    imagearray = numpy.array(imagearray, dtype='uint8')
    return imagearray

imagearray = label_to_color(set_y, color_table)
imagearray = imagearray.reshape(
    (im_width-width)/step_width + 1,  # first image dimension (vertical)
    (im_height-height)/step_height + 1, # second image dimension (horizontal)
    3
)
image = Image.fromarray(imagearray)
imshow(image, cmap=cm.gray)
show()