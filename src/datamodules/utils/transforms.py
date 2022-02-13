from torchvision import transforms
from PIL import Image


def load_image_and_transform(img_path):
    img = Image.open(img_path)
    transform = transforms.ToTensor()
    return transform(img)
