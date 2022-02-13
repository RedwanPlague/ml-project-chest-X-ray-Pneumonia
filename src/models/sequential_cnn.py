import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchinfo import summary
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification.f_beta import F1Score

from .utils.layers import make_layer


class SequentialCNN(pl.LightningModule):
    def __init__(
        self,
        architecture_file='architecture.txt',
        input_channel=1,
        learning_rate=1e-3,
        reg_lambda=1e-3
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        layers = []
        with open(architecture_file) as f:
            for line in f:
                layer, input_channel = make_layer(line, input_channel)
                layers.append(layer)
        self.layers = nn.Sequential(*layers)

        metrics = {
            'Acc': Accuracy(),
            'F1': F1Score(num_classes=3)
        }
        self.train_metrics = MetricCollection(metrics, prefix='train')
        self.val_metrics = MetricCollection(metrics, prefix='val')
        self.test_metrics = MetricCollection(metrics, prefix='test')

    def forward(self, x):
        y = self.layers(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        self.log('trainLoss', loss, prog_bar=False, on_epoch=True, on_step=False)

        metrics = self.train_metrics(logits, y.argmax(dim=-1))
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        self.log('valLoss', loss, prog_bar=True, on_epoch=True, on_step=False)

        metrics = self.val_metrics(logits, y.argmax(dim=-1))
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        self.log('testLoss', loss, prog_bar=True, on_epoch=True, on_step=False)

        metrics = self.test_metrics(logits, y.argmax(dim=-1))
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.reg_lambda
        )


def main():
    model = SequentialCNN(input_channel=3)
    print(model)
    summary(model, input_size=(32, 3, 32, 32), col_names=(
                # "input_size",
                "output_size",
                "kernel_size",
                # "num_params",
                # "mult_adds",
            ))


if __name__ == '__main__':
    main()
