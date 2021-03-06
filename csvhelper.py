import csv


class csvReader:
    def __init__(self, filename, delimiter='\t'):
        self.csvfile = open(filename)
        self.reader = csv.DictReader(self.csvfile, delimiter=delimiter)
        self.__table()
        self.__count = 0

    def next(self):
        row = self.table[self.__count]
        self.__count += 1
        return row

    @property
    def fieldnames(self):
        return self.reader.fieldnames

    def __table(self):
        self.table = []
        for row in self.reader:
            self.table.extend([row])


class csvWriter:
    def __init__(self, filename, fieldnames, delimiter='\t'):
        self.csvfile = open(filename, 'w')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=fieldnames, delimiter=delimiter, lineterminator='\n')
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

    reader = csvReader(filename)

    for row in reader.table[:100]:
        # Load image
        from images import get_image
        image = get_image(row, "google")

        # Convert image into numpy array
        from scipy.misc import fromimage
        img = fromimage(image)

        # Add numpy array to images array
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
    reader = csvReader("nwt-data/Staedte_Deutschland_zufall.csv")
    fieldnames = reader.fieldnames

    xpos = []
    ypos = []

    for row in reader.table:
        X_Coordinate = row['X_Coordinate']
        X_float = float(X_Coordinate)
        xpos.extend([X_float])

        Y_Coordinate = row['Y_Coordinate']
        Y_float = float(Y_Coordinate)
        ypos.extend([Y_float])

        if Y_float < 45:
            print row

        if len(xpos) != len(ypos):
            print row, len(xpos), len(ypos)

    print len(xpos)
    print len(ypos)

    from visualize import point_plot
    point_plot(xpos, ypos)


if __name__ == "__main__":
    main()
