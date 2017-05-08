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
from images import download_image


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
        self.canvas = Canvas(master)
        self.canvas.pack(side=BOTTOM)
        self.show_next_image()

        # OptionsMenu
        self.var = StringVar(master)
        self.var.set("one")  # default value
        self.menu = OptionMenu(frame, self.var, "one", "two", "three")
        self.menu.pack(side=RIGHT)

        # Buttons
        self.accept = Button(frame, text="Accept", command=self.accept, width=10)
        self.accept.pack(side=LEFT)

    def accept(self):
        self.result('Solar', self.var.get())

    def result(self, header, result):
        self.row[header] = result
        self.csvwriter.writerow(self.row)
        self.row = self.csvreader.next()
        self.show_next_image()

    def change_image(self, filepath):
        # Load image
        pilImage = Image.open(filepath)

        # Change canvas size
        self.canvas.config(width=pilImage.width, height=pilImage.height)
        self.imagesprite = self.canvas.create_image(pilImage.width / 2, pilImage.height / 2)

        # Show image on canvas
        self.image = ImageTk.PhotoImage(pilImage)
        self.canvas.itemconfig(self.imagesprite, image=self.image)
        return self.imagesprite

    def show_next_image(self):
        CENTERMODE = "xy"
        filepath = "images/google-{}/{}/{}.jpg".format(CENTERMODE, self.row['ZipCode'].zfill(5), self.row['House_ID'])
        img = None
        while img is None:
            try:
                # Change image
                img = self.change_image(filepath)
            except IOError:
                # Download image from GoogleMaps API
                download_image(filepath, self.row, centermode=CENTERMODE)


def main():
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
