import yaml
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from models.food_squeezenet import FoodSqueezenet
from models.food_resnet18 import FoodResnet18
from trainer.food_trainer import FoodTrainer
from datamodules.food_module import Food101DataModule


def main():
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if config['base_model'] == 'squeezenet':
        model = FoodSqueezenet()
    elif config['base_model'] == 'resnet18':
        model = FoodResnet18()
    else:
        raise ValueError("Invalid base model specified in config.yaml.")

    trainer_model = FoodTrainer(model, config['learning_rate'])

    data_module = Food101DataModule(config_file='./config/config.yaml')

    logger = TensorBoardLogger('logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints',
                                          save_last=True)

    trainer = Trainer(
        max_epochs=config['epochs'],
        gpus=1,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
    )

    trainer.fit(trainer_model, datamodule=data_module)

    torch.save(trainer_model.model.state_dict(),
               f"{config['base_model']}_checkpoint.pt")
