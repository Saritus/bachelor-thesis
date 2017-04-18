# Load csv file
import csv

csvfile = open('nwt-data/Gebaeude_Dresden.csv')
reader = csv.DictReader(csvfile, delimiter='\t')
