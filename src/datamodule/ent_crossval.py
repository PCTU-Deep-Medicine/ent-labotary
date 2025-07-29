import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold

class EntDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
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
    

class CrossValENTDataModule(LightningDataModule):
    def __init__(
            self,
            csv_file: str,
            image_dir: str,
            batch_size: int = 32,
            num_workers: int = 4,
            image_size: int = 224,
            k_folds: int = 5,
            current_fold: int = 0,
            seed: int = 42
    ):
                 
        super().__init__()
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.k_folds = k_folds
        self.current_fold = current_fold
        self.seed = seed

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_file)
        assert {'filename', 'type', 'label'}.issubset(df.columns)

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        dataset = EntDataset(
            dataframe=df,
            image_dir=self.image_dir,
            transform=transform
        )

        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        indices = list(skf.split(df['filename'], df['label']))[self.current_fold]
        
        train_index, val_index = indices
        self.train_dataset = Subset(dataset, train_index)
        self.val_dataset = Subset(dataset, val_index)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )