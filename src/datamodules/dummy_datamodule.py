from pytorch_lightning import LightningDataModule


class DummyDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage=None):
        return super().setup(stage)

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()