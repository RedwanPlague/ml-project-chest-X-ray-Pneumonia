from models.sequential_fcn import SequentialFCN
from datamodules.simple_datamodule import SimpleDataModule
from pytorch_lightning import Trainer


model = SequentialFCN()
dm = SimpleDataModule()
trainer = Trainer(fast_dev_run=True)

trainer.fit(model, dm)