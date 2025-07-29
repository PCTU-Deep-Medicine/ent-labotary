import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger

import matplotlib.pyplot as plt
import numpy as np
import wandb

from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC,
    Specificity,
    MulticlassConfusionMatrix,
    MulticlassROC,
)


class BaseModule(pl.LightningModule):
    def __init__(self, encoder, loss, num_classes: int = 10, class_names=None):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "loss"])
        self.encoder = encoder
        self.loss_fn = loss

        self.NUM_CLASSES = int(num_classes)
        self.class_names = class_names if class_names is not None else [str(i) for i in range(self.NUM_CLASSES)]

        # --- Metrics (macro + per-class) ---
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.NUM_CLASSES, average="macro")

        self.val_precision = Precision(task="multiclass", num_classes=self.NUM_CLASSES, average="macro")
        self.val_precision_pc = Precision(task="multiclass", num_classes=self.NUM_CLASSES, average=None)

        self.val_recall = Recall(task="multiclass", num_classes=self.NUM_CLASSES, average="macro")
        self.val_recall_pc = Recall(task="multiclass", num_classes=self.NUM_CLASSES, average=None)

        self.val_f1 = F1Score(task="multiclass", num_classes=self.NUM_CLASSES, average="macro")
        self.val_f1_pc = F1Score(task="multiclass", num_classes=self.NUM_CLASSES, average=None)

        self.val_specificity = Specificity(task="multiclass", num_classes=self.NUM_CLASSES, average="macro")
        self.val_specificity_pc = Specificity(task="multiclass", num_classes=self.NUM_CLASSES, average=None)

        self.val_auroc = AUROC(task="multiclass", num_classes=self.NUM_CLASSES, average="macro")
        self.val_auroc_pc = AUROC(task="multiclass", num_classes=self.NUM_CLASSES, average=None)

        # Để vẽ ROC curve (lấy FPR/TPR cho từng lớp)
        self.val_roc = MulticlassROC(num_classes=self.NUM_CLASSES)

        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.NUM_CLASSES)

        self.train_loss_epoch = []
        self.val_loss_epoch = []

    def forward(self, x):
        return self.encoder(x)

    # ----------------- TRAIN -----------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.train_loss_epoch.append(loss.detach())
        return loss

    def on_train_epoch_end(self):
        if len(self.train_loss_epoch) > 0:
            avg_train_loss = torch.stack(self.train_loss_epoch).mean()
            self.log("train/loss", avg_train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.train_loss_epoch.clear()

    # ----------------- VAL -----------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        loss = self.loss_fn(logits, y)

        # Update metrics
        self.val_accuracy.update(preds, y)

        self.val_precision.update(preds, y)
        self.val_precision_pc.update(preds, y)

        self.val_recall.update(preds, y)
        self.val_recall_pc.update(preds, y)

        self.val_f1.update(preds, y)
        self.val_f1_pc.update(preds, y)

        self.val_specificity.update(preds, y)
        self.val_specificity_pc.update(preds, y)

        self.val_auroc.update(probs, y)
        self.val_auroc_pc.update(probs, y)

        self.val_roc.update(probs, y)  # dùng để vẽ ROC curve

        self.val_confusion_matrix.update(preds, y)
        self.val_loss_epoch.append(loss.detach())

    def on_validation_epoch_end(self):
        # ---- Compute once (epoch-level) ----
        avg_val_loss = torch.stack(self.val_loss_epoch).mean() if len(self.val_loss_epoch) > 0 else torch.tensor(0.0, device=self.device)

        acc = self.val_accuracy.compute().detach().cpu().item()

        prec = self.val_precision.compute().detach().cpu().item()
        rec = self.val_recall.compute().detach().cpu().item()
        f1 = self.val_f1.compute().detach().cpu().item()
        spec = self.val_specificity.compute().detach().cpu().item()
        auroc_macro = self.val_auroc.compute().detach().cpu().item()

        prec_pc = self.val_precision_pc.compute().detach().cpu().tolist()
        rec_pc = self.val_recall_pc.compute().detach().cpu().tolist()
        f1_pc = self.val_f1_pc.compute().detach().cpu().tolist()
        spec_pc = self.val_specificity_pc.compute().detach().cpu().tolist()
        auroc_pc = self.val_auroc_pc.compute().detach().cpu().tolist()

        # ---- Log macro metrics (epoch) ----
        self.log("val/loss", avg_val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/precision", prec, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/recall", rec, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/specificity", spec, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/auroc", auroc_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # ---- Log per-class metrics (epoch) ----
        for i in range(self.NUM_CLASSES):
            self.log(f"val_pc/precision_pc_{i}", float(prec_pc[i]), on_step=False, on_epoch=True, logger=True)
            self.log(f"val_pc/recall_pc_{i}", float(rec_pc[i]), on_step=False, on_epoch=True, logger=True)
            self.log(f"val_pc/f1_pc_{i}", float(f1_pc[i]), on_step=False, on_epoch=True, logger=True)
            self.log(f"val_pc/specificity_pc_{i}", float(spec_pc[i]), on_step=False, on_epoch=True, logger=True)
            self.log(f"val_pc/auroc_pc_{i}", float(auroc_pc[i]), on_step=False, on_epoch=True, logger=True)

        # ---- Confusion Matrix (image) ----
        cm = self.val_confusion_matrix.compute().detach().cpu().numpy()

        fig_cm = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        im = ax.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix (epoch {self.current_epoch})")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(self.NUM_CLASSES))
        ax.set_yticks(np.arange(self.NUM_CLASSES))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)

        # Annotate counts
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(self.NUM_CLASSES):
            for j in range(self.NUM_CLASSES):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
        plt.tight_layout()

        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({"val/confusion_matrix": wandb.Image(fig_cm)}, commit=False)
        plt.close(fig_cm)

        # ---- ROC curve (per-class) ----
        # fprs, tprs, thresholds are lists with length = num_classes
        fprs, tprs, _ = self.val_roc.compute()
        fprs = [f.cpu().numpy() for f in fprs]
        tprs = [t.cpu().numpy() for t in tprs]

        fig_roc = plt.figure(figsize=(6, 5))
        for i in range(self.NUM_CLASSES):
            plt.plot(fprs[i], tprs[i], label=f"{self.class_names[i]} (AUC={auroc_pc[i]:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves (epoch {self.current_epoch})")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()

        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({"val/roc_curves": wandb.Image(fig_roc)}, commit=True)
        plt.close(fig_roc)

        # ---- Reset for next epoch ----
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_precision_pc.reset()
        self.val_recall.reset()
        self.val_recall_pc.reset()
        self.val_f1.reset()
        self.val_f1_pc.reset()
        self.val_specificity.reset()
        self.val_specificity_pc.reset()
        self.val_auroc.reset()
        self.val_auroc_pc.reset()
        self.val_roc.reset()
        self.val_confusion_matrix.reset()
        self.val_loss_epoch.clear()

    def configure_optimizers(self):
        # Nếu bạn muốn nhận optimizer từ config Hydra:
        # return instantiate(self.hparams.optimizer, params=self.parameters())
        return torch.optim.Adam(self.parameters(), lr=1e-3)
