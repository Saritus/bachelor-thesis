def center_crop_image(path, new_width, new_height):
    from PIL import Image

    im = Image.open(path).convert('RGB')
    width, height = im.size  # Get dimensions

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return im.crop((left, top, right, bottom))


def ensure_dir(filepath):
    import os
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return filepath


def load_image(filepath, size=None):
    from PIL import Image
    image = Image.open(filepath).convert('RGB')
    if size:
        image = image.resize(size, Image.ANTIALIAS)

    from scipy.misc import fromimage
    return fromimage(image)


def download_image(filepath, row, size=(640, 640), zoom=19, maptype="satellite", imageformat="png"):
    x = row['X_Coordinate'].replace(',', '.')
    y = row['Y_Coordinate'].replace(',', '.')

    import urllib
    urlpath = "http://maps.google.com/maps/api/staticmap"
    urlpath += "?center=" + y + "," + x
    urlpath += "&size=" + size[0] + "x" + size[1]
    urlpath += "&zoom=" + zoom
    urlpath += "&maptype=" + maptype
    urlpath += "&format=" + imageformat
    urlpath += "&key=AIzaSyC9d7-JkZseVB_YW9bdIAaFCbQRLTKGaNY"
    urllib.urlretrieve(urlpath, filepath)

    cutoff = 44 if size[0] >= 181 else 32
    image = center_crop_image(filepath, size[0] - cutoff, size[1] - cutoff)
    image.save(filepath)


def main():
    return


if __name__ == "__main__":
    main()
