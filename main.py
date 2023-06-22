import yaml
import torch
from pathlib import Path
# import ssl

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from models.food_squeezenet import FoodSqueezenet
from models.food_resnet18 import FoodResnet18
from trainer.food_trainer import FoodTrainer
from datamodules.food_module import Food101DataModule


def main():
    """ UNSAFE!!! REMOVE IF DATASET WAS DOWNLOADED """
    # ssl._create_default_https_context = ssl._create_unverified_context

    config_path = Path('./config/config.yaml')
    with config_path.open('r') as file:
        config = yaml.safe_load(file)

    if config['base_model'] == 'squeezenet':
        model = FoodSqueezenet(config['pretrained'])
    elif config['base_model'] == 'resnet18':
        model = FoodResnet18(config['pretrained'])
    else:
        raise ValueError("Invalid base model specified in config.yaml.")

    if config['use_saved_model']:
        model_save_path = Path(config['saved_model_path'])
        if model_save_path.exists():
            model.load_state_dict(torch.load(model_save_path))
        else:
            raise FileNotFoundError(
                f"Saved model path not found: {model_save_path}")

    data_module = Food101DataModule(config_file=config_path)

    trainer_model = FoodTrainer(model, config['learning_rate'], data_module)

    logger = TensorBoardLogger('logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints',
                                          save_last=True)

    trainer = Trainer(
        max_epochs=config['epochs'],
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
    )
    trainer.fit(trainer_model, datamodule=data_module)

    model_save_path = Path(f"./models/saved/{config['base_model']}_trained.pt")
    with model_save_path.open('wb') as f:
        torch.save(trainer_model.model.state_dict(), f)


if __name__ == "__main__":
    main()
