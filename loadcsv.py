from functions import center_crop_image, ensure_dir, download_image, load_image


def load_csv(filename):
    X_meta = []
    Y_train = []
    X_images = []
    Coordinates = []

    import csv
    csvfile = open(filename)
    reader = csv.DictReader(csvfile, delimiter='\t')

    count = 0
    for row in reader:
        if count > 10000:
            break
        count += 1

        filepath = "nwt-data/google-images/" + row['ZipCode'].zfill(5) + "/" + row['House_ID'] + ".jpg"
        ensure_dir(filepath)

        # Fill image input array
        try:
            img = load_image(filepath, 300, 300)
        except IOError:
            # Download image from GoogleMaps API
            download_image(filepath, row)
            print count
            img = load_image(filepath, 300, 300)
        X_images.extend([img])

        # Fill coordinates array
        coordinate = [
            float(row['X_Coordinate'].replace(',', '.')),
            float(row['Y_Coordinate'].replace(',', '.')),
        ]
        Coordinates.extend([coordinate])

        # Fill metadata input array
        x = [
            # float(row['X_Coordinate'].replace(',', '.')),
            # float(row['Y_Coordinate'].replace(',', '.')),
            float(hash(row['District'])) % 100,
            float(hash(row['Street'])) % 100,
            # float(row['ZipCode']),
        ]
        X_meta.extend([x])

        # Fill output array
        y = [
            int(row['ZipCode']),
        ]
        Y_train.extend([y])

    import numpy

    Coordinates = numpy.asarray(Coordinates, dtype=numpy.float32)
    X_meta = numpy.asarray(X_meta, dtype=numpy.float32)
    X_images = numpy.asarray(X_images, dtype=numpy.float32)
    Y_train = numpy.asarray(Y_train, dtype=numpy.float32)
    X_train = [X_images, X_meta]

    print ("X_images.shape", X_images.shape)
    print ("Y_train.shape", Y_train.shape)

    return Coordinates, (X_train, Y_train)


def main():
    import csv

    with open('nwt-data/Gebaeude_Dresden.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        xcoords = []
        ycoords = []
        zipcodes = []
        flags = []

        for row in reader:
            xcoords.extend([float(row['X_Coordinate'].replace(',', '.'))])
            ycoords.extend([float(row['Y_Coordinate'].replace(',', '.'))])
            zipcodes.extend([float(hash(row['ZipCode'])) % 17])
            flags.extend([float(row['Flag_Coordinates'])])

    from visualize import point_plot
    point_plot(xcoords, ycoords, zipcodes)


if __name__ == "__main__":
    main()
