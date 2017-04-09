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


def download_image(filepath, row):
    import urllib
    from PIL import Image
    urlpath = "http://maps.google.com/maps/api/staticmap?center="
    urlpath += row['Y_Coordinate'].replace(',', '.') + "," + row['X_Coordinate'].replace(',', '.')
    urlpath += "&zoom=18&size=442x442&maptype=satellite"
    urlpath += "&key=AIzaSyC9d7-JkZseVB_YW9bdIAaFCbQRLTKGaNY"
    urllib.urlretrieve(urlpath, filepath)
    image = center_crop_image(filepath, 400, 400)
    image = image.resize((100, 100), Image.ANTIALIAS)
    image.save(filepath)


def main():
    return


if __name__ == "__main__":
    main()
