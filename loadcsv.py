import csv
with open('data/Gebaeude_Dresden.csv') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    xcoords = []
    ycoords = []
    zipcodes = []
    
    for row in reader:
        
        xcoords.extend([float(row['X_Coordinate'].replace(',','.'))])
        ycoords.extend([float(row['Y_Coordinate'].replace(',','.'))])
        zipcodes.extend([float(row['ZipCode'])])

import matplotlib.pyplot as plt
plt.plot(xcoords, ycoords, 'bo', markersize=0.5)
plt.show()