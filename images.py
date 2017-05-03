def load_image(filepath, size=None):
    from PIL import Image
    image = Image.open(filepath).convert('RGB')
    if size:
        image = image.resize(size, Image.ANTIALIAS)

    from scipy.misc import fromimage
    return fromimage(image)


def combine_images(imagepaths, gridsize):
    from PIL import Image

    images = []
    for filepath in imagepaths:
        images.extend([Image.open(filepath)])

    output = Image.new("RGB", (images[0].width * gridsize[0], images[0].height * gridsize[1]))

    for y in range(0, gridsize[1]):
        for x in range(0, gridsize[0]):
            output.paste(images[y * gridsize[0] + x], (images[0].width * x, images[0].height * y))

    return output
