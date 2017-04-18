# Load csv file
import csv

csvfile = open('nwt-data/Gebaeude_Dresden.csv')
reader = csv.DictReader(csvfile, delimiter='\t')
class csvReader:
    def __init__(self, filename):
        self.csvfile = open(filename)
        self.reader = csv.DictReader(self.csvfile, delimiter='\t')

    def next(self):
        row = self.reader.next()
        return row


class csvWriter:
    def __init__(self, filename):
        fieldnames = ['first_name', 'last_name']
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


class App:
    def __init__(self, master):
        # Frame
        frame = Frame(master)
        frame.pack()

        # Image
        self.canvas = Canvas(root, width=598, height=598)
        self.canvas.pack(side=BOTTOM)
        self.pilImage = Image.open("nwt-data/google-images/01067/100000010.jpg")
        self.image = ImageTk.PhotoImage(self.pilImage)
        self.imagesprite = self.canvas.create_image(299, 299, image=self.image)

        # Buttons
        self.button_yes = Button(frame, text="YES", command=self.yes, width=10)
        self.button_yes.pack(side=LEFT)
        self.button_no = Button(frame, text="NO", command=self.no, width=10)
        self.button_no.pack(side=RIGHT)

        # CSV
        self.csvreader = csvReader('nwt-data/Gebaeude_Dresden.csv')
        self.csvwriter = csvWriter('nwt-data/Output.csv')

    def yes(self):
        print("YES")
        self.change_image("nwt-data/google-images/01067/100000182.jpg")

    def no(self):
        print("NO")
        self.change_image("nwt-data/google-images/01067/100000291.jpg")

    def change_image(self, filepath):
        self.pilImage = Image.open(filepath)
        self.image = ImageTk.PhotoImage(self.pilImage)
        self.canvas.itemconfig(self.imagesprite, image=self.image)


root = Tk()
app = App(root)
root.mainloop()
