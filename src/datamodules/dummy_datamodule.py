import os
from pytorch_lightning import LightningDataModule
from utils.transforms import load_image_and_transform, label_transform


DIR = 'dummy_data'
POS = 'PNEUMONIA'
NEG = 'NORMAL'


def get_data(stage):
    pos_list = os.listdir(os.path.join(DIR, stage, POS))
    neg_list = os.listdir(os.path.join(DIR, stage, NEG))
    x = pos_list + neg_list
    y = ([1] * len(pos_list)) + ([0] * len(neg_list))
    return x, y


class DummyDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.x_train, self.y_train = get_data('train')
        elif stage in ('validate', None):
            self.x_train, self.y_train = get_data('val')
        elif stage in ('test', None):
            self.x_train, self.y_train = get_data('test')

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()