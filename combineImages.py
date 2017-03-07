from PIL import Image

images = []
images.extend([Image.open("samples/1.png").convert('RGB')])
images.extend([Image.open("samples/2.png").convert('RGB')])
images.extend([Image.open("samples/3.png").convert('RGB')])
images.extend([Image.open("samples/4.png").convert('RGB')])
