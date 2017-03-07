from PIL import Image
from pylab import imshow, show, cm, imsave

def combine_images(images):
    if(images[0].size[1] is not images[1].size[1]):
        raise ValueError("Height of image 1 and 2 are different")
    if(images[0].size[0] is not images[2].size[0]):
        raise ValueError("Width of image 1 and 3 are different")
    if(images[1].size[0] is not images[3].size[0]):
        raise ValueError("Width of image 2 and 4 are different")
    if(images[2].size[1] is not images[3].size[1]):
        raise ValueError("Height of image 3 and 4 are different")
    output = Image.new("RGB", (images[0].size[0]+images[1].size[0], images[0].size[1]+images[2].size[1]))
    output.paste(images[0], (0,0))
    output.paste(images[1], (images[0].size[0],0))
    output.paste(images[2], (0,images[0].size[1]))
    output.paste(images[3], (images[0].size[0], images[0].size[1]))
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
