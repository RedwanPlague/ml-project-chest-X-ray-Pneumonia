from pytorch_lightning import Trainer
from torchinfo import summary
import json
import csv
import os
import shutil

from models.sequential_cnn import SequentialCNN
from datamodules.simple_datamodule import SimpleDataModule


with open('parameters.json') as f:
    params = json.load(f)

model = SequentialCNN(
    architecture_file=params['archi_file'],
    input_shape=(1, 1000, 700),
    learning_rate=params['learning_rate'],
    reg_lambda=params['lambda']
)
print(model)
summary(
    model,
    input_size=(params['batch_size'], 1, 1000, 700),
    col_names=("output_size", "kernel_size",)
)

dm = SimpleDataModule(batch_size=params['batch_size'])

trainer = Trainer(
    max_epochs=params['max_epochs'],
    # gpus=-1,
    benchmark=True,
)

trainer.fit(model, dm)

metrics = trainer.callback_metrics
for key in metrics.keys():
    metrics[key] = f'{float(metrics[key]):.5f}'
print(json.dumps(metrics, indent=4))

version = trainer.logger.version

archi_dir = 'archs'
if not os.path.isdir(archi_dir):
    os.makedirs(archi_dir)
shutil.copy(params['archi_file'], os.path.join(archi_dir, f'arch_v{version}.txt'))

del params['archi_file']

data_size = len(dm.train_data)

log = {'version': version, 'data_size': data_size, **params, **metrics}
log_file = 'logs.csv'

if not os.path.isfile(log_file):
    with open(log_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=log.keys())
        writer.writeheader()

with open(log_file, 'a') as f:
    writer = csv.DictWriter(f, fieldnames=log.keys())
    writer.writerow(log)
