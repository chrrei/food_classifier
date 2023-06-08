import pytorch_lightning as pl
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
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        Food101(root='data/', download=True)

    def setup(self, stage=None):
        food101_full = Food101(root='data/', transform=self.transform)
        train_size = int(len(food101_full) * self.train_split)
        val_size = len(food101_full) - train_size

        manual_seed(self.seed)
        self.food101_train, self.food101_val = random_split(food101_full, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.food101_train, batch_size=64, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.food101_val, batch_size=64, shuffle=False, num_workers=4)
