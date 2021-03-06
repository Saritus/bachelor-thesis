# Load csv file

from Tkinter import *
from PIL import ImageTk

from csvhelper import csvReader, csvWriter


class App:
    def __init__(self, master, header, options):
        # Frame
        frame = Frame(master)
        frame.pack()

        # CSV
        self.csvreader = csvReader('images/infopunks_v2/data.csv', delimiter=',')
        fieldnames = self.csvreader.fieldnames
        fieldnames.append('Solar')
        self.csvwriter = csvWriter('images/infopunks_v2/Output.csv', fieldnames, delimiter=',')
        self.row = self.csvreader.next()

        # Image
        self.canvas = Canvas(master)
        self.canvas.pack(side=BOTTOM)
        self.change_image("bing")

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
        self.change_image("bing")

    def change_image(self, api, centermode="address"):
        # Get image
        from images import get_image
        pilImage = get_image(self.row, api, centermode)

        # Change canvas size
        self.canvas.config(width=pilImage.width, height=pilImage.height)
        self.imagesprite = self.canvas.create_image(pilImage.width / 2, pilImage.height / 2)

        # Show image on canvas
        self.image = ImageTk.PhotoImage(pilImage)
        self.canvas.itemconfig(self.imagesprite, image=self.image)
        return self.imagesprite


def main():
    root = Tk()
    options = ["Yes", "No"]
    App(root, 'Solar', options)
    root.mainloop()


if __name__ == "__main__":
    main()
