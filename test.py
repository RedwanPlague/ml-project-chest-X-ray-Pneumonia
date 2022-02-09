import cv2
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
    hs, ws = [], []

    for img_file in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, img_file))
        h, w, c = img.shape
        assert c == 3
        hs.append(h)
        ws.append(w)

    print(f'{min(hs)} {max(hs)} {sum(hs)/len(hs)}')
    print(f'{min(ws)} {max(ws)} {sum(ws)/len(ws)}')
