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

import matplotlib.pyplot as plt

plt.scatter(xcoords, ycoords, c=zipcodes, s=0.5, cmap='jet')
plt.show()


def main():
    pass


if __name__ == "__main__":
    main()
