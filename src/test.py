from torchvision import transforms
from PIL import Image
import os
import torch
from torch import nn
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification.f_beta import F1Score
import torch.nn.functional as F

pred = torch.FloatTensor([
    [-10, -2, -3],
    [10, -2, -3],
    [-10, -2, 3],
])
labels = torch.LongTensor([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
])
# print(F.cross_entropy(pred, labels))

metrics = MetricCollection([
    Accuracy(),
    F1Score(num_classes=3)
])

print(metrics(pred, labels.argmax(dim=-1)))


