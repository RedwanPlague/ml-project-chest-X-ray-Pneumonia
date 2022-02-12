from PIL import Image
import os


dirs = [
    'chest_xray/train/NORMAL',
    'chest_xray/train/PNEUMONIA',
    'chest_xray/val/NORMAL',
    'chest_xray/val/PNEUMONIA',
    'chest_xray/test/NORMAL',
    'chest_xray/test/PNEUMONIA',
]

rgb_cnt, l_cnt = 0, 0

for dir in dirs:
    # hs, ws = [], []
    for img_file in os.listdir(dir):
        img_path = os.path.join(dir, img_file)
        img = Image.open(img_path)
        assert img.mode == 'L'
    # print(f'{min(hs)} {max(hs)} {sum(hs)/len(hs)}')
    # print(f'{min(ws)} {max(ws)} {sum(ws)/len(ws)}')

print(rgb_cnt, l_cnt)