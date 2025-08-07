from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Module, ModuleDict
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    F1Score,
    MulticlassConfusionMatrix,
    MulticlassROC,
    Precision,
    Recall,
    Specificity,
)

import wandb

matplotlib.use("Agg")  # Use non-interactive backend for plotting


class MetricsManager(Module):
    """
    Gom tất cả metric; cho phép log macro & per-class.
    Tham số:
        • num_classes : số lớp phân loại
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        # ────────── định nghĩa metric ──────────
        self.metrics = ModuleDict(
            {
                "accuracy": Accuracy(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "precision": Precision(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "precision_pc": Precision(
                    task="multiclass", num_classes=num_classes, average=None
                ),
                "recall": Recall(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "recall_pc": Recall(
                    task="multiclass", num_classes=num_classes, average=None
                ),
                "f1": F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "f1_pc": F1Score(
                    task="multiclass", num_classes=num_classes, average=None
                ),
                "specificity": Specificity(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "specificity_pc": Specificity(
                    task="multiclass", num_classes=num_classes, average=None
                ),
                "auroc": AUROC(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "auroc_pc": AUROC(
                    task="multiclass", num_classes=num_classes, average=None
                ),
                "roc": MulticlassROC(num_classes=num_classes),
                "conf_matrix": MulticlassConfusionMatrix(num_classes=num_classes),
            }
        )

        self.fold_results = defaultdict(list)  # dùng nếu muốn khái quát nhiều lần train

    # ───────────────────────── update ─────────────────────────
    def update(self, preds, probs, targets):
        m = self.metrics
        m.accuracy.update(preds, targets)
        m.precision.update(preds, targets)
        m.precision_pc.update(preds, targets)
        m.recall.update(preds, targets)
        m.recall_pc.update(preds, targets)
        m.f1.update(preds, targets)
        m.f1_pc.update(preds, targets)
        m.specificity.update(preds, targets)
        m.specificity_pc.update(preds, targets)
        m.auroc.update(probs, targets)
        m.auroc_pc.update(probs, targets)
        m.roc.update(probs, targets)
        m.conf_matrix.update(preds, targets)

    # ───────────────────────── compute ────────────────────────
    def _macro_dict(self):
        m = self.metrics
        return {
            "accuracy": m.accuracy.compute().cpu().item(),
            "precision": m.precision.compute().cpu().item(),
            "recall": m.recall.compute().cpu().item(),
            "f1": m.f1.compute().cpu().item(),
            "specificity": m.specificity.compute().cpu().item(),
            "auroc": m.auroc.compute().cpu().item(),
        }

    # ─────────── compute & log (val / test) ───────────
    def compute_and_log(
        self,
        logger: WandbLogger | None,
        epoch: int,
        phase: str = "val",
        log_fn=None,
        include_plots: bool = False,
    ):
        """
        phase : 'val' hoặc 'test'
        include_plots : True => log ROC & Confusion-Matrix
        """
        macro = self._macro_dict()

        # per-class list
        prec_pc = self.metrics.precision_pc.compute().cpu().tolist()
        rec_pc = self.metrics.recall_pc.compute().cpu().tolist()
        f1_pc = self.metrics.f1_pc.compute().cpu().tolist()
        spec_pc = self.metrics.specificity_pc.compute().cpu().tolist()
        auc_pc = self.metrics.auroc_pc.compute().cpu().tolist()

        # ─── log bằng Lightning self.log ───
        if log_fn:
            for k, v in macro.items():
                log_fn(f"{phase}/macro/{k}", v, prog_bar=(k == "f1"), on_epoch=True)

        # ─── log vào wandb ───
        if logger:
            # macro
            logger.log_metrics({f"{phase}/macro/{k}": v for k, v in macro.items()})
            # per-class
            for i in range(self.num_classes):
                logger.log_metrics(
                    {
                        f"{phase}/per_class/precision_{i}": float(prec_pc[i]),
                        f"{phase}/per_class/recall_{i}": float(rec_pc[i]),
                        f"{phase}/per_class/f1_{i}": float(f1_pc[i]),
                        f"{phase}/per_class/specificity_{i}": float(spec_pc[i]),
                        f"{phase}/per_class/auroc_{i}": float(auc_pc[i]),
                    }
                )

        # ─── Chỉ vẽ hình ở test ───
        if include_plots and logger:
            self._log_confusion_matrix(logger, epoch, phase)
            self._log_roc_curve(logger, epoch, phase, auc_pc)

        self.reset()

    # ───────────────────────── plots ─────────────────────────
    def _log_confusion_matrix(self, logger, epoch, phase):
        cm = self.metrics.conf_matrix.compute().cpu().numpy()
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        im = ax.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix ({phase}, epoch {epoch})")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )
        plt.tight_layout()
        logger.experiment.log(
            {f"{phase}/confusion_matrix": wandb.Image(fig)}, commit=False
        )
        plt.close(fig)

    def _log_roc_curve(self, logger, epoch, phase, auc_pc):
        fprs, tprs, _ = self.metrics.roc.compute()
        fprs = [f.cpu().numpy() for f in fprs]
        tprs = [t.cpu().numpy() for t in tprs]

        fig = plt.figure(figsize=(6, 5))
        for i in range(self.num_classes):
            plt.plot(fprs[i], tprs[i], label=f"Class {i} (AUC={auc_pc[i]:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves ({phase}, epoch {epoch})")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()

        logger.experiment.log({f"{phase}/roc_curves": wandb.Image(fig)}, commit=True)
        plt.close(fig)

    # ───────────────────────── reset ─────────────────────────
    def reset(self):
        for m in self.metrics.values():
            m.reset()
