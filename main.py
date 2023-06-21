import yaml
import torch
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from models.food_squeezenet import FoodSqueezenet
from models.food_resnet18 import FoodResnet18
from trainer.food_trainer import FoodTrainer
from datamodules.food_module import Food101DataModule


def main():
    config_path = Path('./config/config.yaml')
    with config_path.open('r') as file:
        config = yaml.safe_load(file)

    if config['base_model'] == 'squeezenet':
        model = FoodSqueezenet()
    elif config['base_model'] == 'resnet18':
        model = FoodResnet18()
    else:
        raise ValueError("Invalid base model specified in config.yaml.")

    trainer_model = FoodTrainer(model, config['learning_rate'])

    data_module = Food101DataModule(config_file=config_path)

    logger = TensorBoardLogger('logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints',
                                          save_last=True)

    gpus = 1 if torch.cuda.is_available() and config['use_gpu'] else None
    trainer = Trainer(
        max_epochs=config['epochs'],
        gpus=gpus,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
    )

    trainer.fit(trainer_model, datamodule=data_module)

    model_save_path = Path(f"./models/saved/{config['base_model']}_trained.pt")
    torch.save(trainer_model.model.state_dict(), model_save_path)


if __name__ == "__main__":
    main()
