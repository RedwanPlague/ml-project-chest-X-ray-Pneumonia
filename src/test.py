from tkinter import Image
from torchvision import transforms
from PIL import Image


img_path = 'dummy_data/train/PNEUMONIA/person2_bacteria_3.jpeg'

img = Image.open(img_path)
print(img.size)

resize = transforms.Resize((1500, 2000))
im2 = resize(img)
print(im2.size)

im2.save(f'haha.{img.format.lower()}')