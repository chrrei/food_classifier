import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models


class FoodSqueezenet(pl.LightningModule):
    def __init__(self, pretrained):
        super().__init__()
        self.num_classes = 101  # 101 classes for Food101
        self.model = models.squeezenet1_1(pretrained=pretrained)
        num_ftrs = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        return self.model(x)
