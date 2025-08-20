import pytorch_lightning as pl
import torch
import torch.nn as nn

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

        feature_dim = self.encoder.get_classifier().in_features
        self.encoder.reset_classifier(0)  # reset classifier to get feature dimension

        if hasattr(self.encoder, "fc"):
            self.encoder.fc = nn.Linear(feature_dim, num_classes)
        elif hasattr(self.encoder, "head"):
            self.encoder.head = nn.Linear(feature_dim, num_classes)
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
        # 1) Chọn LR theo batch size
        global_bs = 64  # hoặc tự tính: batch_size_per_gpu * num_gpus * grad_accum
        base_lr = 1e-4
        lr = base_lr * max(global_bs / 256.0, 0.5)  # sàn 0.5 cho batch nhỏ
        lr = min(lr, 1e-4)  # trần để tránh quá cao khi batch lớn

        opt = torch.optim.AdamW(
            self.parameters(),
            lr=lr,  # vd ~3e-5 → 1e-4 tuỳ batch
            weight_decay=1e-2,
            betas=(0.9, 0.999),
        )

        # 2) Warmup 5% epochs (ví dụ 15/300)
        warmup_epochs = max(int(0.05 * self.max_epochs), 5)
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, total_iters=warmup_epochs
        )

        # 3) Cosine cho phần còn lại (đi về eta_min)
        remain_epochs = max(self.max_epochs - warmup_epochs, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=remain_epochs, eta_min=1e-6
        )

        # 4) Ghép lại
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }
