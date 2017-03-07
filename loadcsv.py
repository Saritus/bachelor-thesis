import csv
with open('data/Gebaeude_Dresden.csv') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    xcoords = []
    ycoords = []
    
    for row in reader:
        
        xcoords.extend([float(row['SP_GEOMETRY_Center'].replace(',','.'))])
        ycoords.extend([float(row['X_Coordinate'].replace(',','.'))])

import matplotlib.pyplot as plt
plt.plot(xcoords, ycoords, 'bo', markersize=0.5)
plt.show()
