# Load csv file
import csv


class csvReader:
    def __init__(self, filename):
        self.csvfile = open(filename)
        self.reader = csv.DictReader(self.csvfile, delimiter='\t')

    def next(self):
        row = self.reader.next()
        return row

    def fieldnames(self):
        return self.reader.fieldnames


class csvWriter:
    def __init__(self, filename, fieldnames):
        self.csvfile = open(filename, 'w')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=fieldnames, delimiter='\t')
        self.writer.writeheader()

    def writerow(self, row):
        self.writer.writerow(row)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


from Tkinter import *
from PIL import Image, ImageTk
from functions import download_image


class App:
    def __init__(self, master):
        # Frame
        frame = Frame(master)
        frame.pack()

        # CSV
        self.csvreader = csvReader('nwt-data/Gebaeude_Dresden_shuffle.csv')
        fieldnames = self.csvreader.fieldnames()
        fieldnames.append('Solar')
        self.csvwriter = csvWriter('nwt-data/Output.csv', fieldnames)
        self.row = self.csvreader.next()

        # Image
        self.canvas = Canvas(root, width=598, height=598)
        self.canvas.pack(side=BOTTOM)
        self.imagesprite = self.canvas.create_image(299, 299)
        self.show_next_image()

        # Buttons
        self.button_yes = Button(frame, text="YES", command=self.yes, width=10)
        self.button_yes.pack(side=LEFT)
        self.button_no = Button(frame, text="NO", command=self.no, width=10)
        self.button_no.pack(side=RIGHT)

    def yes(self):
        self.row['Solar'] = 1
        self.csvwriter.writerow(self.row)
        self.row = self.csvreader.next()
        self.show_next_image()

    def no(self):
        self.row['Solar'] = 0
        self.csvwriter.writerow(self.row)
        self.row = self.csvreader.next()
        self.show_next_image()

    def change_image(self, filepath):
        self.pilImage = Image.open(filepath)
        self.image = ImageTk.PhotoImage(self.pilImage)
        self.canvas.itemconfig(self.imagesprite, image=self.image)
        return self.imagesprite

    def show_next_image(self):
        filepath = "nwt-data/google-images/" + self.row['ZipCode'].zfill(5) + "/" + self.row['House_ID'] + ".jpg"
        img = None
        while img is None:
            try:
                # Change image
                img = self.change_image(filepath)
            except IOError:
                # Download image from GoogleMaps API
                download_image(filepath, self.row)


root = Tk()
app = App(root)
root.mainloop()
