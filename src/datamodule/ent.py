import os

import pandas as pd
from PIL import Image
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class EntDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)
        return image, label


class SplitENTDataModule(LightningDataModule):
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 224,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

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
        df = pd.read_csv(self.csv_file)
        assert {'filename', 'label_type', 'label'}.issubset(df.columns)

        # Tính tỉ lệ cho từng phần
        test_size = self.test_ratio
        val_size = self.val_ratio / (1 - test_size)

        train_val_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            stratify=df['label'],
            random_state=self.seed,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=df.loc[train_val_idx, 'label'],
            random_state=self.seed,
        )

        train_tf, eval_tf = self._build_transforms()

        # Tạo 3 dataset riêng với transform riêng
        self.train_dataset = EntDataset(
            dataframe=df.loc[train_idx], image_dir=self.image_dir, transform=train_tf
        )
        self.val_dataset = EntDataset(
            dataframe=df.loc[val_idx], image_dir=self.image_dir, transform=eval_tf
        )
        self.test_dataset = EntDataset(
            dataframe=df.loc[test_idx], image_dir=self.image_dir, transform=eval_tf
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
