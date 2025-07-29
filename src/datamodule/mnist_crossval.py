# mmist datamodule for cross-validation
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
from PIL import Image

class CrossvalMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, image_size: int = 224, batch_size: int = 32, num_workers: int = 4, num_folds: int = 5, current_fold: int = 0, seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_folds = num_folds
        self.current_fold = current_fold
        self.image_size = image_size
        self.seed = seed

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load MNIST dataset
        mnist_train = MNIST(self.data_dir, train=True, download=True, transform=transform)
        mnist_test = MNIST(self.data_dir, train=False, download=True, transform=transform)

        # Combine train and test for cross-validation
        full_dataset = mnist_train + mnist_test

        # Create stratified folds
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        indices = list(range(len(full_dataset)))
        labels = [y for _, y in full_dataset]

        fold_indices = list(skf.split(indices, labels))
        
        # Get indices for the current fold
        train_indices, val_indices = fold_indices[self.current_fold]
        
        # Create subsets for training and validation
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)