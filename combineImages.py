from PIL import Image
from pylab import imshow, show, cm, imsave

def combine_images(images):
    width = images[0].size[0]
    height = images[0].size[1]
    output = Image.new("RGB", (width*2, height*2))
    output.paste(images[0], (0,0))
    output.paste(images[1], (width,0))
    output.paste(images[2], (0,height))
    output.paste(images[3], (width, height))
    return output

images = []
images.extend([Image.open("samples/1.png").convert('RGB')])
images.extend([Image.open("samples/2.png").convert('RGB')])
images.extend([Image.open("samples/3.png").convert('RGB')])
images.extend([Image.open("samples/4.png").convert('RGB')])

output = combine_images(images)
imshow(output)
show()

output.save("samples/output.png")
