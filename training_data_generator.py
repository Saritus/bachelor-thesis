# Load csv file
import csv

csvfile = open('nwt-data/Gebaeude_Dresden.csv')
reader = csv.DictReader(csvfile, delimiter='\t')

from Tkinter import *
from PIL import Image, ImageTk


class App:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()

        self.canvas = Canvas(root, width=598, height=598)
        self.canvas.pack(side=BOTTOM)
        self.pilImage = Image.open("nwt-data/google-images/01067/100000010.jpg")
        self.image = ImageTk.PhotoImage(self.pilImage)
        self.imagesprite = self.canvas.create_image(299, 299, image=self.image)

        self.button = Button(frame, text="QUIT", command=quit)
        self.button.pack(side=LEFT)
        self.slogan = Button(frame, text="Hello", command=self.write_slogan)
        self.slogan.pack(side=LEFT)

    def write_slogan(self):
        print("Tkinter is easy to use!")


root = Tk()
app = App(root)
root.mainloop()
