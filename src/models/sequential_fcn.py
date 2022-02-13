import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchinfo import summary

from .utils.layers import make_layer


class SequentialFCN(pl.LightningModule):
    def __init__(
        self,
        architecture_file='src/models/architecture.txt',
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

    def forward(self, x):
        y = self.layers(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y_act = batch
        y_pred = self(x)

        loss = F.cross_entropy(y_act, y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_act = batch
        y_pred = self(x)

        loss = F.cross_entropy(y_act, y_pred)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_act = batch
        y_pred = self(x)

        loss = F.cross_entropy(y_act, y_pred)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.reg_lambda
        )


def main():
    model = SequentialFCN(input_channel=3)
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
