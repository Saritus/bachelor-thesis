# Load csv file
import csv

csvfile = open('nwt-data/Gebaeude_Dresden.csv')
reader = csv.DictReader(csvfile, delimiter='\t')
class csvReader:
    def __init__(self, filename):
        self.csvfile = open(filename)
        self.reader = csv.DictReader(self.csvfile)

    def next(self):
        row = self.reader.next()
        return row


class csvWriter:
    def __init__(self, filename):
        self.csvfile = open(filename)
        self.writer = csv.DictWriter(self.csvfile)
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
        self.button = Button(frame, text="YES", command=self.yes, width=10)
        self.button.pack(side=LEFT)
        self.slogan = Button(frame, text="NO", command=self.no, width=10)
        self.slogan.pack(side=RIGHT)

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
