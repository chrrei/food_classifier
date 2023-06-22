import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models


class FoodResnet18(pl.LightningModule):
    def __init__(self, pretrained):
        super().__init__()
        self.num_classes = 101  # 101 classes for Food101
        self.model = models.resnet18(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        return self.model(x)
