from PIL import Image
from PIL import ImageChops
import numpy as np
import os


dirs = [
    'dummy_data/train/NORMAL',
    'dummy_data/train/PNEUMONIA',
    'dummy_data/val/NORMAL',
    'dummy_data/val/PNEUMONIA',
    'dummy_data/test/NORMAL',
    'dummy_data/test/PNEUMONIA',
]

for dir in dirs:
    # hs, ws = [], []

    for img_file in os.listdir(dir):
        img = Image.open(os.path.join(dir, img_file))
        if len(img.getbands()) == 3:
            print('-------------')
            img_arr = np.array(img).transpose((2, 0, 1))
            print(img_arr.shape)
            print(np.isclose(img_arr[0], img_arr[1]).all())
            print(np.isclose(img_arr[0], img_arr[2]).all())
            print(np.isclose(img_arr[1], img_arr[2]).all())
            im2 = img.convert('L')
            im2_arr = np.array(im2)
            print(im2_arr.shape)
            print(np.isclose(im2_arr, img_arr[1]).all())
            print(np.isclose(im2_arr, img_arr[2]).all())
            print(np.isclose(im2_arr, img_arr[2]).all())
            print('-------------')
            # im2 = Image.open('check.jpeg').convert('RGB')
            # if ImageChops.difference(img, im2).getbbox():
            #     print('diff')
            # else:
            #     print('same')
            # assert False
        # h, w, c = img.size
        # assert c == 3
        # hs.append(h)
        # ws.append(w)

    # print(f'{min(hs)} {max(hs)} {sum(hs)/len(hs)}')
    # print(f'{min(ws)} {max(ws)} {sum(ws)/len(ws)}')
