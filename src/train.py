from models.sequential_fcn import SequentialFCN
from datamodules.simple_datamodule import SimpleDataModule
from pytorch_lightning import Trainer
from torchinfo import summary


model = SequentialFCN(learning_rate=1e-4)
print(model)
summary(model, input_size=(32, 1, 1000, 700), col_names=(
                # "input_size",
                "output_size",
                "kernel_size",
                # "num_params",
                # "mult_adds",
            ))


dm = SimpleDataModule(batch_size=32)
trainer = Trainer(max_epochs=5)

trainer.fit(model, dm)

print(trainer.callback_metrics)