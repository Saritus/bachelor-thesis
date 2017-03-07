from PIL import Image
from pylab import imshow, show, cm, imsave

def combine_images(images):
    output = Image.new("RGB", (100, 100))
    output.paste(images[0], (0,0,50,50))
    output.paste(images[1], (50,0,100,50))
    output.paste(images[2], (0,50,50,100))
    output.paste(images[3], (50,50,100,100))
    return output

images = []
images.extend([Image.open("samples/1.png").convert('RGB')])
images.extend([Image.open("samples/2.png").convert('RGB')])
images.extend([Image.open("samples/3.png").convert('RGB')])
images.extend([Image.open("samples/4.png").convert('RGB')])

output = combine_images(images)
imshow(output)
show()
