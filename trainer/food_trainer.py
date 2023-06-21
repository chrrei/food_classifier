import torch
import torchmetrics
import pytorch_lightning as pl


class FoodTrainer(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

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
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = self.val_accuracy(torch.nn.functional.softmax(y_hat, dim=1), y)
        self.log('val_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('val_acc',
                 acc,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

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
