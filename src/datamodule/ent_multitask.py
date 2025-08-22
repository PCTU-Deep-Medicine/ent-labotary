import os
from typing import Dict, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

__all__ = ["ENTMultiTaskDataModule"]


class _FineToMultiTaskWrapper(Dataset):
    """Wrap an ImageFolder (fine-grained) dataset to output (image, (label_type, label)).

    label_type mapping (fixed):
        0-4  -> 0 (ear)
        5-8  -> 1 (nose)
        9-11 -> 2 (throat)
    """

    def __init__(self, base: datasets.ImageFolder):
        self.base = base

    def __len__(self):  # pragma: no cover - simple forwarder
        return len(self.base)

    @staticmethod
    def _label_to_type(label: torch.Tensor | int) -> int:
        if isinstance(label, torch.Tensor):
            label_int = int(label.item())
        else:
            label_int = int(label)
        if 0 <= label_int <= 4:
            return 0
        if 5 <= label_int <= 8:
            return 1
        if 9 <= label_int <= 11:
            return 2
        raise ValueError(
            f"Label {label_int} outside expected 0..11 range for coarse mapping"
        )

    def __getitem__(self, idx: int):
        img, fine_label = self.base[idx]
        coarse = self._label_to_type(fine_label)
        return img, (
            torch.tensor(coarse, dtype=torch.long),
            torch.tensor(fine_label, dtype=torch.long),
        )

    @property
    def class_to_idx(self):  # fine mapping
        return self.base.class_to_idx


class ENTMultiTaskDataModule(LightningDataModule):
    """Multi-task DataModule for ENT dataset.

    Expects the fine-grained ImageFolder directory root produced by `split_multitask.py`:
        root/fine/train/CLASS_NAME/*
        root/fine/val/CLASS_NAME/*
        root/fine/test/CLASS_NAME/*

    Returns each batch item as (image, (label_type, label)) so that the MultiTaskModule
    can unpack it directly.
    """

    def __init__(
        self,
        multitask_root: str = "data/12endo_multitask/fine",  # path pointing to the 'fine' branch
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 224,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.multitask_root = multitask_root.rstrip("/")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.class_to_idx: Optional[Dict[str, int]] = None
        self.num_fine_classes: Optional[int] = None
        self.num_coarse_classes: int = 3  # ear, nose, throat (fixed)

    # --------------------------- transforms ---------------------------
    def _build_transforms(self):
        train_tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02
                        )
                    ],
                    p=0.5,
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return train_tf, eval_tf

    # ------------------------------ setup ------------------------------
    def setup(self, stage: str | None = None):  # pragma: no cover - simple determinism
        train_dir = os.path.join(self.multitask_root, "train")
        val_dir = os.path.join(self.multitask_root, "val")
        test_dir = os.path.join(self.multitask_root, "test")

        train_tf, eval_tf = self._build_transforms()

        base_train = datasets.ImageFolder(train_dir, transform=train_tf)
        base_val = datasets.ImageFolder(val_dir, transform=eval_tf)
        base_test = datasets.ImageFolder(test_dir, transform=eval_tf)

        self.train_dataset = _FineToMultiTaskWrapper(base_train)
        self.val_dataset = _FineToMultiTaskWrapper(base_val)
        self.test_dataset = _FineToMultiTaskWrapper(base_test)

        self.class_to_idx = base_train.class_to_idx  # fine mapping
        self.num_fine_classes = len(self.class_to_idx)

        assert (
            base_val.class_to_idx == self.class_to_idx
        ), "class_to_idx của val khác train"
        assert (
            base_test.class_to_idx == self.class_to_idx
        ), "class_to_idx của test khác train"

    # ---------------------------- dataloaders -------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
