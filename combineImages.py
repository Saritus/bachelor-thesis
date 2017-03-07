from PIL import Image
from pylab import imshow, show, cm, imsave

def combine_images(images):
    # Checking for errors
    if(images[0].height is not images[1].height):
        raise ValueError("Height of image 1 and 2 are different")
    if(images[0].width is not images[2].width):
        raise ValueError("Width of image 1 and 3 are different")
    if(images[1].width is not images[3].width):
        raise ValueError("Width of image 2 and 4 are different")
    if(images[2].height is not images[3].height):
        raise ValueError("Height of image 3 and 4 are different")

    # Create new image big enough for the 4 images
    output = Image.new("RGB", (images[0].width+images[1].width,
                               images[0].height+images[2].height))

    # Put the images at the right position
    output.paste(images[0], (0,0))
    output.paste(images[1], (images[0].width,0))
    output.paste(images[2], (0,images[0].height))
    output.paste(images[3], (images[0].width, images[0].height))
    return output

# Open the images
images = []
images.extend([Image.open("samples/1.png").convert('RGB')])
images.extend([Image.open("samples/2.png").convert('RGB')])
images.extend([Image.open("samples/3.png").convert('RGB')])
images.extend([Image.open("samples/4.png").convert('RGB')])

output = combine_images(images)

# Show the combined picture
imshow(output)
show()

# Save the combined picture
output.save("samples/output.png")
