# Load csv file
import csv

csvfile = open('nwt-data/Gebaeude_Dresden.csv')
reader = csv.DictReader(csvfile, delimiter='\t')

from Tkinter import *


class App:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        self.button = Button(frame, text="QUIT", command=quit)
        self.button.pack(side=LEFT)
        self.slogan = Button(frame, text="Hello", command=self.write_slogan)
        self.slogan.pack(side=LEFT)

    def write_slogan(self):
        print("Tkinter is easy to use!")


root = Tk()
app = App(root)
root.mainloop()
