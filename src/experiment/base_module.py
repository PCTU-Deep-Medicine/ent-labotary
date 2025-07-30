import pytorch_lightning as pl
import torch

from src.utils.metrics import MetricsManager


class BaseModule(pl.LightningModule):
    def __init__(self, encoder, loss, num_classes: int = 10, fold_id: int = 0):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "loss"])
        self.fold_id = fold_id
        self.encoder = encoder
        self.loss_fn = loss
        self.metrics = MetricsManager(num_classes, fold_id=fold_id)

        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.train_losses.append(loss)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log(
            f"train/loss_fold{self.fold_id}", avg_loss, on_epoch=True, prog_bar=True
        )
        self.train_losses.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        loss = self.loss_fn(logits, y)
        self.metrics.update(preds, probs, y)
        self.val_losses.append(loss)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log(f"val/loss_fold{self.fold_id}", avg_loss, on_epoch=True, prog_bar=True)
        self.val_losses.clear()

        # Log metrics cho checkpoint monitor
        self.metrics.compute_and_log(self.logger, self.current_epoch, log_fn=self.log)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
