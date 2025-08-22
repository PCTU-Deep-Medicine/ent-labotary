from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger

from .metrics import MetricsManager


@dataclass
class TaskOutputs:
    """Container cho các tensor dự đoán của 1 task.
    (Không bắt buộc ở mức Lightning, chỉ để gọn code khi truyền vào update.)
    """

    preds: any  # torch.Tensor shape (B,)
    probs: any  # torch.Tensor shape (B, C)
    targets: any  # torch.Tensor shape (B,)


class MultiTaskMetricsManager(nn.Module):
    """
    Quản lý 2 bộ metric độc lập cho bài toán multi-task (label_type & label).

    Sử dụng 2 instance của `MetricsManager` (tái dùng logic plotting & logging).

    Cách dùng tối thiểu trong LightningModule:
        self.mt_metrics = MultiTaskMetricsManager(num_classes_type=A, num_classes_label=B)
        ... trong validation/test step:
            self.mt_metrics.update(
                task_type=TaskOutputs(preds_type, probs_type, y_type),
                task_label=TaskOutputs(preds_label, probs_label, y_label),
            )
        ... trong on_validation_epoch_end():
            self.mt_metrics.compute_and_log(logger=self.logger, epoch=self.current_epoch, phase_prefix="val")
        ... trong on_test_epoch_end():
            self.mt_metrics.compute_and_log(logger=self.logger, epoch=self.current_epoch, phase_prefix="test", include_plots=True)
    """

    def __init__(
        self,
        num_classes_type: int,
        num_classes_label: int,
        use_slash: bool = True,
        enable_joint: bool = False,
        joint_mode: str = "full",  # 'full' (T*L classes) or 'combined12' (fine-like, penalize coarse sai)
    ):
        """
        use_slash: True => phase pattern "val/label_type" (hierarchical like single-task)
                    False => phase pattern "val_label_type" (flat like previous version)
        """
        super().__init__()
        # Register as submodules so Lightning moves them to device
        self.type_metrics = MetricsManager(num_classes=num_classes_type)
        self.label_metrics = MetricsManager(num_classes=num_classes_label)
        self.use_slash = use_slash
        self.enable_joint = enable_joint
        self.joint_mode = joint_mode
        self.num_classes_type = num_classes_type
        self.num_classes_label = num_classes_label
        if enable_joint:
            if joint_mode == "full":
                self.joint_metrics = MetricsManager(
                    num_classes=num_classes_type * num_classes_label
                )
            elif joint_mode == "combined12":
                # 12 lớp giống single-task fine
                self.joint_metrics = MetricsManager(num_classes=num_classes_label)
            else:
                raise ValueError(f"joint_mode không hỗ trợ: {joint_mode}")
        else:
            self.joint_metrics = None

    # ------------------------------------------------------------------
    def update(self, task_type: TaskOutputs, task_label: TaskOutputs):
        self.type_metrics.update(task_type.preds, task_type.probs, task_type.targets)
        self.label_metrics.update(
            task_label.preds, task_label.probs, task_label.targets
        )

        if self.enable_joint and self.joint_metrics is not None:
            if self.joint_mode == "full":
                # Joint target index = type * N_label + label
                joint_targets = (
                    task_type.targets * self.num_classes_label + task_label.targets
                )
                pt = task_type.probs  # (B, T)
                pl = task_label.probs  # (B, L)
                joint_probs = torch.einsum("bt,bl->btl", pt, pl).reshape(pt.size(0), -1)
                joint_preds = (
                    task_type.preds * self.num_classes_label + task_label.preds
                )
                self.joint_metrics.update(joint_preds, joint_probs, joint_targets)
            elif self.joint_mode == "combined12":
                # Dùng 12 lớp như single-task, nhưng 1 prediction chỉ được coi đúng khi cả coarse & fine đúng.
                # Nếu fine đúng nhưng coarse sai → ta bắt nó thành sai bằng cách đổi predicted class sang lớp khác.
                preds_label = task_label.preds.clone()
                targets_label = task_label.targets
                probs_label = task_label.probs
                # coarse correctness
                coarse_ok = task_type.preds == task_type.targets
                fine_ok = preds_label == targets_label
                mask_fix = (~coarse_ok) & fine_ok  # các mẫu fine đúng nhưng coarse sai
                if mask_fix.any():
                    # chuyển sang 1 lớp khác để torchmetrics tính sai (chọn lớp (y+1)%C)
                    preds_label[mask_fix] = (
                        targets_label[mask_fix] + 1
                    ) % self.num_classes_label
                self.joint_metrics.update(preds_label, probs_label, targets_label)
            else:
                raise ValueError(f"joint_mode không hỗ trợ: {self.joint_mode}")

    # ------------------------------------------------------------------
    def compute_and_log(
        self,
        logger: Optional[WandbLogger],
        epoch: int,
        phase_prefix: str,
        log_fn=None,
        include_plots: bool = False,
    ):
        """Gọi compute_and_log cho từng task.

        phase_prefix: ví dụ "val" => sẽ log "val/label_type", "val/label" (hoặc dạng underscore nếu use_slash=False)
        include_plots: chỉ nên True ở test
        """
        sep = "/" if self.use_slash else "_"
        phase_type = f"{phase_prefix}{sep}label_type"
        phase_label = f"{phase_prefix}{sep}label"

        # label_type
        self.type_metrics.compute_and_log(
            logger=logger,
            epoch=epoch,
            phase=phase_type,
            log_fn=log_fn,
            include_plots=include_plots,
        )
        # label
        self.label_metrics.compute_and_log(
            logger=logger,
            epoch=epoch,
            phase=phase_label,
            log_fn=log_fn,
            include_plots=include_plots,
        )
        # joint
        if self.enable_joint and self.joint_metrics is not None:
            phase_joint = f"{phase_prefix}{sep}joint"
            # Only plot confusion/ROC for joint if plots requested (can be large)
            self.joint_metrics.compute_and_log(
                logger=logger,
                epoch=epoch,
                phase=phase_joint,
                log_fn=log_fn,
                include_plots=include_plots,
            )

    # ------------------------------------------------------------------
    def reset(self):
        self.type_metrics.reset()
        self.label_metrics.reset()
        if self.enable_joint and self.joint_metrics is not None:
            self.joint_metrics.reset()


__all__ = ["MultiTaskMetricsManager", "TaskOutputs"]
