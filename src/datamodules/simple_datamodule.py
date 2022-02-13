import os
import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from .utils.transforms import load_image_and_transform


DIR = 'dummy_data'
POS = 'PNEUMONIA'
NEG = 'NORMAL'
WORKERS = 4


def get_data(stage):
    pos_dir = os.path.join(DIR, stage, POS)
    neg_dir = os.path.join(DIR, stage, NEG)
    pos_list = [os.path.join(pos_dir, file) for file in os.listdir(pos_dir)]
    neg_list = [os.path.join(neg_dir, file) for file in os.listdir(neg_dir)]
    img_file_names = pos_list + neg_list
    labels = ([1] * len(pos_list)) + ([0] * len(neg_list))
    return img_file_names, labels


def custom_collate(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    y = torch.FloatTensor(y)
    return [x, y]


class CustomDataset(Dataset):
    def __init__(self, img_file_names, labels):
        super().__init__()
        self.x = [load_image_and_transform(img) for img in img_file_names]
        self.y = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


class SimpleDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.train_data = CustomDataset(*get_data('train'))
            self.val_data = CustomDataset(*get_data('val'))
        elif stage in ('validate', None):
            self.val_data = CustomDataset(*get_data('val'))
        elif stage in ('test', None):
            self.test_data = CustomDataset(*get_data('test'))

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=WORKERS,
            collate_fn=custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=WORKERS,
            collate_fn=custom_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=WORKERS,
            collate_fn=custom_collate
        )