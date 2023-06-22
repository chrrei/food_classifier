import pytorch_lightning as pl
import numpy as np
import yaml
from torch import manual_seed
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import Food101
from torchvision import transforms


class Food101DataModule(pl.LightningDataModule):
    def __init__(self, config_file):
        super().__init__()

        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        self.seed = self.config['seed']
        self.train_split = self.config['train_split']
        self.augmentation = self.config['augmentation']
        self.batches = self.config['batch_size']

        self.transform_train, self.transform_val = self._preprocess_data()

    def _preprocess_data(self):
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]

        if self.augmentation:
            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-10, 10)),
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4, hue=0.1),
            ]
            transform_train = transforms.Compose(train_transforms +
                                                 base_transforms)
        else:
            transform_train = transforms.Compose(base_transforms)

        transform_val = transforms.Compose(base_transforms)

        return transform_train, transform_val

    def prepare_data(self):
        Food101(root='data/', download=True)

    def setup(self, stage=None):
        food101_full = Food101(root='data/', transform=self.transform_train)
        train_size = int(len(food101_full) * self.train_split)
        val_size = len(food101_full) - train_size

        manual_seed(self.seed)
        np.random.seed(self.seed)
        self.food101_train, self.food101_val = \
            random_split(food101_full, [train_size, val_size])

        self.food101_val.dataset.transform = self.transform_val

    def train_dataloader(self):
        return DataLoader(self.food101_train, batch_size=self.batches,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.food101_val, batch_size=self.batches,
                          shuffle=False, num_workers=4)
