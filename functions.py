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
    # TODO: replace row with x and y
    # TODO: make switch between google staticmaps and bing maps possible
    import urllib
    from PIL import Image
    urlpath = "http://maps.google.com/maps/api/staticmap?center="
    urlpath += row['Y_Coordinate'].replace(',', '.') + "," + row['X_Coordinate'].replace(',', '.')
    # TODO: use adresse instead of x,y-coordinates
    urlpath += "&zoom=18&size=640x640&maptype=satellite"
    # TODO: download 640x640 images
    urlpath += "&key=AIzaSyC9d7-JkZseVB_YW9bdIAaFCbQRLTKGaNY"
    urllib.urlretrieve(urlpath, filepath)
    image = image.resize((100, 100), Image.ANTIALIAS)
    image = center_crop_image(filepath, 598, 598)
    # TODO: reduce size of images on demand, not right after download
    image.save(filepath)
    # TODO: make switch between satellite map and bird eyes view


def main():
    return


if __name__ == "__main__":
    main()
