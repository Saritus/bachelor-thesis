import csv

from functions import ensure_dir
from images import load_image, download_image


class csvReader:
    def __init__(self, filename, delimiter='\t'):
        self.csvfile = open(filename)
        self.reader = csv.DictReader(self.csvfile, delimiter=delimiter)

    def next(self):
        row = self.reader.next()
        return row

    @property
    def fieldnames(self):
        return self.reader.fieldnames

    @property
    def table(self):
        table = []
        for row in self.reader:
            table.extend([row])
        return table


class csvWriter:
    def __init__(self, filename, fieldnames, delimiter='\t'):
        self.csvfile = open(filename, 'w')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=fieldnames, delimiter=delimiter)
        self.writer.writeheader()

    def writerow(self, row):
        self.writer.writerow(row)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


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

        filepath = "images/google-xy/" + row['ZipCode'].zfill(5) + "/" + row['House_ID'] + ".jpg"
        ensure_dir(filepath)

        # Fill image input array
        img = None
        while img is None:
            try:
                # Load image from hard disk
                img = load_image(filepath, (100, 100))
            except IOError:
                # Download image from GoogleMaps API
                download_image(filepath, row)
                print count
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
    # Reader
    reader = csvReader("nwt-data/Gebaeude_Dresden_shuffle.csv")
    table = reader.table

    writer = csvWriter("nwt-data/Output.csv", reader.fieldnames)

    # load_csv("nwt-data/Gebaeude_Dresden_shuffle.csv")


if __name__ == "__main__":
    main()
