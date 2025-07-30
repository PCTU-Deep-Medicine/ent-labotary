import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    Accuracy, Precision, Recall, F1Score,
    AUROC, Specificity, MulticlassConfusionMatrix, MulticlassROC
)
from torch.nn import Module, ModuleDict
from collections import defaultdict

class MetricsManager(Module):
    def __init__(self, num_classes, fold_id=0):
        super().__init__()
        self.num_classes = num_classes
        self.fold_id = fold_id

        # Metrics
        self.metrics = ModuleDict({
            'val_accuracy': Accuracy(task="multiclass", num_classes=num_classes, average="macro"),
            'val_precision': Precision(task="multiclass", num_classes=num_classes, average="macro"),
            'val_precision_pc': Precision(task="multiclass", num_classes=num_classes, average=None),
            'val_recall': Recall(task="multiclass", num_classes=num_classes, average="macro"),
            'val_recall_pc': Recall(task="multiclass", num_classes=num_classes, average=None),
            'val_f1': F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            'val_f1_pc': F1Score(task="multiclass", num_classes=num_classes, average=None),
            'val_specificity': Specificity(task="multiclass", num_classes=num_classes, average="macro"),
            'val_specificity_pc': Specificity(task="multiclass", num_classes=num_classes, average=None),
            'val_auroc': AUROC(task="multiclass", num_classes=num_classes, average="macro"),
            'val_auroc_pc': AUROC(task="multiclass", num_classes=num_classes, average=None),
            'val_roc': MulticlassROC(num_classes=num_classes),
            'val_confusion_matrix': MulticlassConfusionMatrix(num_classes=num_classes)
        })

        self.val_loss_epoch = []
        self.fold_results = defaultdict(list)

    def update(self, preds, probs, y):
        self.metrics.val_accuracy.update(preds, y)
        self.metrics.val_precision.update(preds, y)
        self.metrics.val_precision_pc.update(preds, y)
        self.metrics.val_recall.update(preds, y)
        self.metrics.val_recall_pc.update(preds, y)
        self.metrics.val_f1.update(preds, y)
        self.metrics.val_f1_pc.update(preds, y)
        self.metrics.val_specificity.update(preds, y)
        self.metrics.val_specificity_pc.update(preds, y)
        self.metrics.val_auroc.update(probs, y)
        self.metrics.val_auroc_pc.update(probs, y)
        self.metrics.val_roc.update(probs, y)
        self.metrics.val_confusion_matrix.update(preds, y)

    def compute(self):
        """Compute macro metrics (dùng cho self.log trong Lightning)."""
        return {
            "accuracy": self.metrics.val_accuracy.compute().cpu().item(),
            "precision": self.metrics.val_precision.compute().cpu().item(),
            "recall": self.metrics.val_recall.compute().cpu().item(),
            "f1": self.metrics.val_f1.compute().cpu().item(),
            "specificity": self.metrics.val_specificity.compute().cpu().item(),
            "auroc": self.metrics.val_auroc.compute().cpu().item(),
        }

    def compute_and_log(self, logger, epoch, log_fn=None):
        # Macro metrics
        macro_metrics = self.compute()

        # Per-class metrics
        prec_pc = self.metrics.val_precision_pc.compute().cpu().tolist()
        rec_pc = self.metrics.val_recall_pc.compute().cpu().tolist()
        f1_pc = self.metrics.val_f1_pc.compute().cpu().tolist()
        spec_pc = self.metrics.val_specificity_pc.compute().cpu().tolist()
        auroc_pc = self.metrics.val_auroc_pc.compute().cpu().tolist()

        # Save fold metrics
        for k, v in macro_metrics.items():
            self.fold_results[f"{k}_macro"].append(v)

        for i in range(self.num_classes):
            self.fold_results[f"precision_pc_{i}"].append(prec_pc[i])
            self.fold_results[f"recall_pc_{i}"].append(rec_pc[i])
            self.fold_results[f"f1_pc_{i}"].append(f1_pc[i])
            self.fold_results[f"specificity_pc_{i}"].append(spec_pc[i])
            self.fold_results[f"auroc_pc_{i}"].append(auroc_pc[i])

        # Log metrics bằng Lightning self.log
        if log_fn:
            for k, v in macro_metrics.items():
                log_fn(f"val/fold_{self.fold_id}/macro/{k}", v, prog_bar=(k == "f1"), on_epoch=True)

        # Log metrics vào Wandb
        if logger:
            logger.log_metrics({
                f"val/fold_{self.fold_id}/macro/{k}": v for k, v in macro_metrics.items()
            })

            for i in range(self.num_classes):
                logger.log_metrics({
                    f"val/fold_{self.fold_id}/per_class/precision_{i}": float(prec_pc[i]),
                    f"val/fold_{self.fold_id}/per_class/recall_{i}": float(rec_pc[i]),
                    f"val/fold_{self.fold_id}/per_class/f1_{i}": float(f1_pc[i]),
                    f"val/fold_{self.fold_id}/per_class/specificity_{i}": float(spec_pc[i]),
                    f"val/fold_{self.fold_id}/per_class/auroc_{i}": float(auroc_pc[i]),
                })

        # Confusion matrix
        self._log_confusion_matrix(logger, epoch)
        # ROC curve
        self._log_roc_curve(logger, auroc_pc, epoch)

        self.reset()

    def _log_confusion_matrix(self, logger, epoch):
        cm = self.metrics.val_confusion_matrix.compute().cpu().numpy()
        fig_cm = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        im = ax.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix (epoch {epoch})")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
        plt.tight_layout()

        if isinstance(logger, WandbLogger):
            logger.experiment.log({f"val/fold_{self.fold_id}/confusion_matrix": wandb.Image(fig_cm)}, commit=False)
        plt.close(fig_cm)

    def _log_roc_curve(self, logger, auroc_pc, epoch):
        fprs, tprs, _ = self.metrics.val_roc.compute()
        fprs = [f.cpu().numpy() for f in fprs]
        tprs = [t.cpu().numpy() for t in tprs]

        fig_roc = plt.figure(figsize=(6, 5))
        for i in range(self.num_classes):
            plt.plot(fprs[i], tprs[i], label=f"Class {i} (AUC={auroc_pc[i]:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves (epoch {epoch})")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()

        if isinstance(logger, WandbLogger):
            logger.experiment.log({f"val/fold_{self.fold_id}/roc_curves": wandb.Image(fig_roc)}, commit=True)
        plt.close(fig_roc)

    def log_all_folds_summary(self, logger):
        """Log mean ± std for all folds"""
        if logger:
            for key, values in self.fold_results.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                if "pc" in key:
                    logger.log_metrics({
                        f"val/all_folds/per_class/{key}_mean": mean_val,
                        f"val/all_folds/per_class/{key}_std": std_val,
                    })
                else:
                    logger.log_metrics({
                        f"val/all_folds/macro/{key}_mean": mean_val,
                        f"val/all_folds/macro/{key}_std": std_val,
                    })

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
        self.val_loss_epoch.clear()
