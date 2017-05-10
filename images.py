from functions import ensure_dir


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


def download_image(filepath, row, size=(640, 640), zoom=20, maptype="satellite", imageformat="png", centermode="xy"):
    ensure_dir(filepath)
    x = row['X_Coordinate'].replace(',', '.')
    y = row['Y_Coordinate'].replace(',', '.')
    address = "{}+{}+{}+{}".format(row['Street'], row['HouseNr'], row['ZipCode'].zfill(5), row['City'])
    apikey = "AIzaSyC9d7-JkZseVB_YW9bdIAaFCbQRLTKGaNY"

    import urllib
    urlpath = "http://maps.google.com/maps/api/staticmap"
    if centermode == "xy":  # Latitude and Longitude
        urlpath += "?center={},{}".format(y, x)
    elif centermode == "address":  # Address
        urlpath += "?center={}".format(address)
    else:
        raise ValueError("Invalid centermode: {}. Use xy or address instead.".format(centermode))
    urlpath += "&size={}x{}".format(size[0], size[1])
    urlpath += "&zoom={}".format(zoom)
    urlpath += "&maptype={}".format(maptype)
    urlpath += "&format={}".format(imageformat)
    urlpath += "&key={}".format(apikey)
    urllib.urlretrieve(urlpath, filepath)

    cutoff = 44 if size[0] >= 181 else 32
    image = center_crop_image(filepath, size[0] - cutoff, size[1] - cutoff)
    image.save(filepath)


def center_crop_image(path, new_width, new_height):
    from PIL import Image

    im = Image.open(path).convert('RGB')
    width, height = im.size  # Get dimensions

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return im.crop((left, top, right, bottom))


def get_image(row, api, centermode="xy"):
    from PIL import Image
    # Google Maps
    if api == "google":
        filepath = "images/google-{}/{}/{}.jpg"
        address = "{}+{}+{}+{}".format(row['Street'], row['HouseNr'],
                                       row['ZipCode'].zfill(5), row['City'])
        filepath = filepath.format(centermode, row['ZipCode'].zfill(5), address)
        pilImage = None
        while pilImage is None:
            try:
                # Open image from filepath
                pilImage = Image.open(filepath)
            except IOError:
                # Download image from GoogleMaps API
                download_image(filepath, row, centermode=centermode)
        return pilImage

    # Bing Maps
    elif api == "bing":
        # Top
        TL = 'images/infopunks_v2/' + row['Top Left Path'] + '.png'
        TC = 'images/infopunks_v2/' + row['Top Center Path'] + '.png'
        TR = 'images/infopunks_v2/' + row['Top Right Path'] + '.png'
        # Middle
        ML = 'images/infopunks_v2/' + row['Middle Left Path'] + '.png'
        MC = 'images/infopunks_v2/' + row['Middle Center Path'] + '.png'
        MR = 'images/infopunks_v2/' + row['Middle Right Path'] + '.png'
        # Bottom
        BL = 'images/infopunks_v2/' + row['Bottom Left Path'] + '.png'
        BC = 'images/infopunks_v2/' + row['Bottom Center Path'] + '.png'
        BR = 'images/infopunks_v2/' + row['Bottom Right Path'] + '.png'

        # Array
        images = [TL, TC, TR, ML, MC, MR, BL, BC, BR]

        # Combine images
        from images import combine_images
        combined_image = combine_images(images, (3, 3))

        return combined_image

    # Unknown API
    else:
        # Invalid API
        raise ValueError("Invalid api: {}. Use google or bing instead.".format(api))


def main():
    return


if __name__ == "__main__":
    main()
