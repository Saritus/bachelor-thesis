# Load csv file
import csv


class csvReader:
    def __init__(self, filename, delimiter='\t'):
        self.csvfile = open(filename)
        self.reader = csv.DictReader(self.csvfile, delimiter=delimiter)

    def next(self):
        row = self.reader.next()
        return row

    def fieldnames(self):
        return self.reader.fieldnames


class csvWriter:
    def __init__(self, filename, fieldnames, delimiter='\t'):
        self.csvfile = open(filename, 'w')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=fieldnames, delimiter=delimiter)
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
    def __init__(self, master, header, options):
        # Frame
        frame = Frame(master)
        frame.pack()

        # CSV
        # self.csvreader = csvReader('nwt-data/Gebaeude_Dresden_shuffle.csv')
        self.csvreader = csvReader('images/infopunks_v2/data.csv', delimiter=',')
        fieldnames = self.csvreader.fieldnames()
        fieldnames.append('Solar')
        # self.csvwriter = csvWriter('nwt-data/Output.csv', fieldnames)
        self.csvwriter = csvWriter('images/infopunks_v2/Output.csv', fieldnames, delimiter=',')
        self.row = self.csvreader.next()

        # Image
        self.canvas = Canvas(master)
        self.canvas.pack(side=BOTTOM)
        self.show_next_image()

        # OptionsMenu
        self.header = header
        self.var = StringVar(master)
        self.var.set(options[0])  # default value
        self.menu = apply(OptionMenu, (frame, self.var) + tuple(options))
        self.menu.pack(side=RIGHT)

        # Buttons
        self.accept = Button(frame, text="Accept", command=self.accept, width=10)
        self.accept.pack(side=LEFT)

    def accept(self):
        self.row[self.header] = self.var.get()
        self.csvwriter.writerow(self.row)
        self.row = self.csvreader.next()
        self.show_next_image()

    def change_image(self, pilImage):
        # Change canvas size
        self.canvas.config(width=pilImage.width, height=pilImage.height)
        self.imagesprite = self.canvas.create_image(pilImage.width / 2, pilImage.height / 2)

        # Show image on canvas
        self.image = ImageTk.PhotoImage(pilImage)
        self.canvas.itemconfig(self.imagesprite, image=self.image)
        return self.imagesprite

    def show_next_image(self):
        # Get image
        img = self.get_image("bing")

        # Change image
        self.change_image(img)

    def get_image(self, api, centermode="xy"):
        # Google Maps
        if api == "google":
            filepath = "images/google-{}/{}/{}.jpg"
            filepath = filepath.format(centermode, self.row['ZipCode'].zfill(5), self.row['House_ID'])
            pilImage = None
            while pilImage is None:
                try:
                    # Open image from filepath
                    pilImage = Image.open(filepath)
                except IOError:
                    # Download image from GoogleMaps API
                    download_image(filepath, self.row, centermode=centermode)
            return pilImage

        # Bing Maps
        elif api == "bing":
            # Top
            TL = self.row['Top Left Path'] + '.png'
            TC = self.row['Top Center Path'] + '.png'
            TR = self.row['Top Right Path'] + '.png'
            # Middle
            ML = self.row['Middle Left Path'] + '.png'
            MC = self.row['Middle Center Path'] + '.png'
            MR = self.row['Middle Right Path'] + '.png'
            # Bottom
            BL = self.row['Bottom Left Path'] + '.png'
            BC = self.row['Bottom Center Path'] + '.png'
            BR = self.row['Bottom Right Path'] + '.png'

            # Array
            images = [TL, TC, TR, ML, MC, MR, BL, BC, BR]

            # Combine images
            from images import combine_images
            combined_image = combine_images(images, (3, 3))

            # Not implemented yet
            raise ValueError("Bing is not implemented yet")

        # Unknown API
        else:
            # Invalid API
            raise ValueError("Invalid api: {}. Use google or bing instead.".format(api))


def main():
    root = Tk()
    options = ["Yes", "No"]
    App(root, 'Solar', options)
    root.mainloop()


if __name__ == "__main__":
    main()
