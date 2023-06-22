import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.models.squeezenet import SqueezeNet1_1_Weights


class FoodSqueezenet(pl.LightningModule):
    def __init__(self, pretrained):
        super().__init__()
        self.num_classes = 101  # 101 classes for Food101
        self.model = models.squeezenet1_1()

        if pretrained:
            state_dict = SqueezeNet1_1_Weights.DEFAULT.get_state_dict(
                progress=True
            )
            self.model.load_state_dict(state_dict)

        input_channels = self.model.classifier[1].in_channels

        self.model.classifier[1] = nn.Conv2d(input_channels,
                                             self.num_classes,
                                             kernel_size=(1, 1),
                                             stride=(1, 1))

    def forward(self, x):
        return self.model(x)
