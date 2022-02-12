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

rgb_cnt, l_cnt = 0, 0

for dir in dirs:
    # hs, ws = [], []
    for img_file in os.listdir(dir):
        img = Image.open(os.path.join(dir, img_file))
        if len(img.getbands()) == 3:
            rgb_cnt += 1
            img_arr = np.array(img).transpose((2, 0, 1))
            r, g, b = img_arr
            diff = np.abs(r - g).sum() + np.abs(g - b).sum() + np.abs(b - r).sum()
            if diff != 0:
                print(diff)
            # im2 = Image.open('check.jpeg').convert('RGB')
            # if ImageChops.difference(img, im2).getbbox():
            #     print('diff')
            # else:
            #     print('same')
            # assert False
        else:
            l_cnt += 1
        # h, w, c = img.size
        # assert c == 3
        # hs.append(h)
        # ws.append(w)

    # print(f'{min(hs)} {max(hs)} {sum(hs)/len(hs)}')
    # print(f'{min(ws)} {max(ws)} {sum(ws)/len(ws)}')

print(rgb_cnt, l_cnt)