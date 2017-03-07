import csv
with open('data/Gebaeude_Dresden.csv') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
        print(row["City"])
