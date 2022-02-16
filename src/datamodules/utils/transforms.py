from torchvision import transforms
from PIL import Image
import torch


def load_image_and_transform(img_path):
    img = Image.open(img_path)
    transform = transforms.ToTensor()
    x = transform(img)
    x = torch.concat((x, x, x))
    # print(x.shape)
    return x


# load_image_and_transform('xray_300_420/train/normal/IM-0003-0001.jpeg')
