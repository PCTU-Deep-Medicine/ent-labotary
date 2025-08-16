import os
from typing import Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ENTDataModule(LightningDataModule):
    def __init__(
        self,
        split_root: str,  # thư mục gốc sau tách: vd "data/12endo"
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 224,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.split_root = split_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # sẽ được gán trong setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_to_idx: Optional[Dict[str, int]] = None
        self.num_classes: Optional[int] = None

    def _build_transforms(self):
        train_tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandAugment(num_ops=3, magnitude=7),
                transforms.ToTensor(),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )
        return train_tf, eval_tf

    def setup(self, stage=None):
        train_dir = os.path.join(self.split_root, "train")
        val_dir = os.path.join(self.split_root, "val")
        test_dir = os.path.join(self.split_root, "test")

        train_tf, eval_tf = self._build_transforms()

        # ImageFolder tự suy ra nhãn theo tên thư mục con
        self.train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
        self.val_dataset = datasets.ImageFolder(val_dir, transform=eval_tf)
        self.test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)

        # Lưu lại mapping và số lớp cho tiện truy cập ở model/LightningModule
        self.class_to_idx = self.train_dataset.class_to_idx
        self.num_classes = len(self.class_to_idx)

        # Đảm bảo val/test dùng cùng mapping như train (thường OK nếu folder đồng nhất)
        assert (
            self.val_dataset.class_to_idx == self.class_to_idx
        ), "class_to_idx của val khác train"
        assert (
            self.test_dataset.class_to_idx == self.class_to_idx
        ), "class_to_idx của test khác train"

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # shuffle ở train
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
