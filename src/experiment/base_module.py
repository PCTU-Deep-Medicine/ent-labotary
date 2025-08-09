import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.utils.metrics import MetricsManager


class BaseModule(pl.LightningModule):
    """
    LightningModule chung cho bài ENT-Endoscopy.
    • Chỉ vẽ Confusion-Matrix & ROC ở phase test.
    """

    def __init__(self, encoder, loss, num_classes: int = 12, max_epochs: int = 100):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "loss"])
        self.encoder = encoder
        self.loss_fn = loss
        self.metrics = MetricsManager(num_classes=num_classes)
        self.max_epochs = max_epochs

        self.train_losses, self.val_losses = [], []

    # ────────────────────────────── forward ──────────────────────────────
    def forward(self, x):
        return self.encoder(x)

    # ─────────────────────────────── train ───────────────────────────────
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_losses.append(loss)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log("train/loss", avg_loss, on_epoch=True, prog_bar=True)
        self.train_losses.clear()

    # ────────────────────────────── validate ─────────────────────────────
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        loss = self.loss_fn(logits, y)

        self.metrics.update(preds, probs, y)
        self.val_losses.append(loss)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log("val/loss", avg_loss, on_epoch=True, prog_bar=True)
        self.val_losses.clear()

        # ‼️ Không log hình ở validation
        self.metrics.compute_and_log(
            logger=self.logger,
            epoch=self.current_epoch,
            phase="val",
            log_fn=self.log,
            include_plots=False,
        )

    # ─────────────────────────────── test ────────────────────────────────
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        loss = self.loss_fn(logits, y)

        self.metrics.update(preds, probs, y)
        self.val_losses.append(loss)  # tái sử dụng list

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log("test/loss", avg_loss, on_epoch=True, prog_bar=True)
        self.val_losses.clear()

        # ✅ Log hình ở test
        self.metrics.compute_and_log(
            logger=self.logger,
            epoch=self.current_epoch,
            phase="test",
            log_fn=self.log,
            include_plots=True,
        )

    # ──────────────────────────── optimizer ──────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        warmup = LinearLR(opt, start_factor=0.1, total_iters=10)
        cosine = CosineAnnealingLR(opt, T_max=self.max_epochs - 10, eta_min=1e-6)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[10])
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
