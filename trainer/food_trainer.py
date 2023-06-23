import torch
import torchmetrics
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


class FoodTrainer(pl.LightningModule):
    def __init__(self, model, lr, datamodule):
        super().__init__()
        self.model = model
        self.lr = lr
        self.datamodule = datamodule
        self.y_hats = []
        self.ys = []
        self.val_losses = []
        self.val_accs = []
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass',
                                                    num_classes=101)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass',
                                                  num_classes=101)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task='multiclass',
            num_classes=101
        )

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = self.train_accuracy(torch.nn.functional.softmax(y_hat, dim=1), y)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('train_acc',
                 acc,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = torch.nn.functional.cross_entropy(y_hat, y)
        val_acc = self.val_accuracy(
            torch.nn.functional.softmax(y_hat, dim=1),
            y
            )
        self.log('val_loss',
                 val_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('val_acc',
                 val_acc,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.y_hats.append(y_hat)
        self.ys.append(y)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)

    def on_validation_epoch_end(self):
        y_hats = torch.cat(self.y_hats)
        ys = torch.cat(self.ys)
        conf_mat = self.confusion_matrix(y_hats.argmax(dim=1), ys)
        val_confusion_matrix = conf_mat.float()
        val_confusion_matrix /= val_confusion_matrix.sum()

        avg_losses = torch.mean(torch.stack(self.val_losses))
        avg_accs = torch.mean(torch.stack(self.val_accs))

        self.log('avg_val_loss',
                 avg_losses,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('avg_val_accs',
                 avg_accs,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(val_confusion_matrix, annot=True, ax=ax, cmap='coolwarm')
        logs_folder = Path("./logs")
        img_filename = f"confusion_matrix_epoch_{self.current_epoch}.png"
        img_filepath = logs_folder / Path(img_filename)
        plt.savefig(img_filepath, format='png')

        # like this?
        # self.log.add_figure("Confusion matrix", fig, self.current_epoch)

        self.y_hats = []
        self.ys = []
        self.val_losses = []
        self.val_accs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = \
            torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=self.lr,
                                                steps_per_epoch=len(
                                                    self.train_dataloader()
                                                ),
                                                epochs=self.trainer.max_epochs)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
