from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.utils.metrics_mutitask import MultiTaskMetricsManager, TaskOutputs

# ----------------------------- batch typing -----------------------------
BatchType = Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (x, y_type, y_label)
    Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # (x, (y_type, y_label))
    Tuple[
        torch.Tensor, Dict[str, torch.Tensor]
    ],  # (x, {"label_type": y_type, "label": y_label})
    Tuple[
        torch.Tensor, torch.Tensor
    ],  # (x, y_label)  -> sẽ tự suy ra y_type từ y_label
]


class MultiTaskModule(pl.LightningModule):
    """
    Multi-task LightningModule (giữ phong cách giống `BaseModule`).
    • Dùng 1 encoder chung + 2 head: label_type & label
    • Chỉ vẽ Confusion-Matrix & ROC ở phase test
    • Log loss trung bình ở cuối mỗi epoch (train/val/test)
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_label_type_classes: int,
        num_label_classes: int,
        loss_fn_type: nn.Module | None = None,
        loss_fn_label: nn.Module | None = None,
        w_label_type: float = 1.0,
        w_label: float = 1.0,
        max_epochs: int = 100,
        enable_joint: bool = True,
        joint_mode: str = "full",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "loss_fn_type", "loss_fn_label"])
        self.encoder = encoder
        self.max_epochs = max_epochs

        # Lấy feature dim (giữ cách reset tương tự base nhưng cần head riêng)
        if hasattr(self.encoder, "reset_classifier"):
            self.encoder.reset_classifier(0)  # bỏ head cũ để lấy embedding
        feat_dim = getattr(self.encoder, "num_features", None)
        if feat_dim is None:
            raise AttributeError(
                "Encoder phải có thuộc tính num_features sau reset_classifier(0)"
            )

        # 2 classification heads
        self.head_type = nn.Linear(feat_dim, num_label_type_classes)
        self.head_label = nn.Linear(feat_dim, num_label_classes)

        # Loss
        self.loss_type = loss_fn_type or nn.CrossEntropyLoss()
        self.loss_label = loss_fn_label or nn.CrossEntropyLoss()
        self.w_label_type = w_label_type
        self.w_label = w_label

        # Metrics (gộp qua quản lý đa nhiệm)
        self.mt_metrics = MultiTaskMetricsManager(
            num_classes_type=num_label_type_classes,
            num_classes_label=num_label_classes,
            use_slash=True,
            enable_joint=enable_joint,
            joint_mode=joint_mode,
        )

        # Accumulate losses giống BaseModule
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []  # dùng riêng thay vì tái sử dụng

    # ------------------------------- forward -----------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x)
        return {
            "logits_type": self.head_type(feats),
            "logits_label": self.head_label(feats),
        }

    # --------------------------- batch unpack ---------------------------
    def _label_to_type(self, y_label: torch.Tensor) -> torch.Tensor:
        """Map fine-grained label (0-11) -> coarse type (ear/nose/throat).

        Mapping (cố định):
            0-4  -> 0 (ear)
            5-8  -> 1 (nose)
            9-11 -> 2 (throat)
        """
        if (y_label < 0).any() or (y_label > 11).any():
            raise ValueError("label ngoài khoảng 0..11 => không map được label_type")
        y_type = torch.zeros_like(y_label)
        y_type[y_label >= 5] = 1
        y_type[y_label >= 9] = 2
        return y_type

    def _unpack_batch(
        self, batch: BatchType
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Allow both tuple and list batches
        if not isinstance(batch, (tuple, list)):
            raise ValueError(f"Batch format không hỗ trợ (type={type(batch)}).")

        # Common case: (x, y_label)
        if len(batch) == 2 and isinstance(batch[1], torch.Tensor):
            x = batch[0]
            y_label = batch[1]
            y_type = self._label_to_type(y_label)
            return x, y_type, y_label

        # Potential case: (x, (y_type, y_label)) where inner may be tuple/list
        if (
            len(batch) == 2
            and isinstance(batch[1], (tuple, list))
            and len(batch[1]) == 2
        ):
            x = batch[0]
            y_type, y_label = batch[1]
            return x, y_type, y_label

        # Potential case: (x, { 'label_type': ..., 'label': ... })
        if len(batch) == 2 and isinstance(batch[1], dict):
            x = batch[0]
            tgt = batch[1]
            y_type = tgt.get("label_type")
            y_label = tgt.get("label")
            if y_type is not None and y_label is not None:
                return x, y_type, y_label

        # Potential case: (x, y_type, y_label)
        if len(batch) == 3 and all(isinstance(b, torch.Tensor) for b in batch[1:]):
            x = batch[0]
            y_type = batch[1]
            y_label = batch[2]
            return x, y_type, y_label

        # Last resort: try to introspect and give a detailed error
        structure = {
            "batch_type": type(batch).__name__,
            "len": len(batch),
            "elem_types": [type(b).__name__ for b in batch],
        }
        # If second element is a sequence, include its inner types
        if len(batch) > 1 and isinstance(batch[1], (tuple, list)):
            structure["second_elem_len"] = len(batch[1])
            structure["second_elem_types"] = [type(x).__name__ for x in batch[1]]
        raise ValueError(f"Batch format không hỗ trợ. Debug info: {structure}")

    # ---------------------------- train step ----------------------------
    def training_step(self, batch: BatchType, batch_idx: int):
        x, y_type, y_label = self._unpack_batch(batch)
        out = self.forward(x)
        lt = self.loss_type(out["logits_type"], y_type)
        ll = self.loss_label(out["logits_label"], y_label)
        total = self.w_label_type * lt + self.w_label * ll
        self.train_losses.append(total.detach())
        return total

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log("train/loss", avg_loss, on_epoch=True, prog_bar=True)
        self.train_losses.clear()

    # --------------------------- validation -----------------------------
    def validation_step(self, batch: BatchType, batch_idx: int):
        x, y_type, y_label = self._unpack_batch(batch)
        out = self.forward(x)
        lt = self.loss_type(out["logits_type"], y_type)
        ll = self.loss_label(out["logits_label"], y_label)
        total = self.w_label_type * lt + self.w_label * ll
        self.val_losses.append(total.detach())

        # metrics update
        probs_type = torch.softmax(out["logits_type"], dim=1)
        preds_type = torch.argmax(probs_type, dim=1)
        probs_label = torch.softmax(out["logits_label"], dim=1)
        preds_label = torch.argmax(probs_label, dim=1)
        self.mt_metrics.update(
            TaskOutputs(preds=preds_type, probs=probs_type, targets=y_type),
            TaskOutputs(preds=preds_label, probs=probs_label, targets=y_label),
        )

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log("val/loss", avg_loss, on_epoch=True, prog_bar=True)
        self.val_losses.clear()

        # Không vẽ hình ở validation
        self.mt_metrics.compute_and_log(
            logger=self.logger,
            epoch=self.current_epoch,
            phase_prefix="val",
            log_fn=self.log,
            include_plots=False,
        )

    # ------------------------------- test -------------------------------
    def test_step(self, batch: BatchType, batch_idx: int):
        x, y_type, y_label = self._unpack_batch(batch)
        out = self.forward(x)
        lt = self.loss_type(out["logits_type"], y_type)
        ll = self.loss_label(out["logits_label"], y_label)
        total = self.w_label_type * lt + self.w_label * ll
        self.test_losses.append(total.detach())

        probs_type = torch.softmax(out["logits_type"], dim=1)
        preds_type = torch.argmax(probs_type, dim=1)
        probs_label = torch.softmax(out["logits_label"], dim=1)
        preds_label = torch.argmax(probs_label, dim=1)
        self.mt_metrics.update(
            TaskOutputs(preds=preds_type, probs=probs_type, targets=y_type),
            TaskOutputs(preds=preds_label, probs=probs_label, targets=y_label),
        )

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        self.log("test/loss", avg_loss, on_epoch=True, prog_bar=True)
        self.test_losses.clear()

        # Vẽ hình ở test
        self.mt_metrics.compute_and_log(
            logger=self.logger,
            epoch=self.current_epoch,
            phase_prefix="test",
            log_fn=self.log,
            include_plots=True,
        )

    # ---------------------- optimizer & scheduler -----------------------
    def configure_optimizers(self):
        global_bs = 64  # có thể sửa theo setup thực tế
        base_lr = 1e-4
        lr = base_lr * max(global_bs / 256.0, 0.5)
        lr = min(lr, 1e-4)

        opt = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.999)
        )
        warmup_epochs = max(int(0.05 * self.max_epochs), 5)
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, total_iters=warmup_epochs
        )
        remain_epochs = max(self.max_epochs - warmup_epochs, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=remain_epochs, eta_min=1e-6
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ------------------------------- utils ------------------------------
    def predict_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0):
        x, _, _ = self._unpack_batch(batch)
        return self.forward(x)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


__all__ = ["MultiTaskModule"]
