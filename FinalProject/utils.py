import os
from pathlib import Path
from typing import Tuple, List, Union
import numpy as np

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

class CassavaDataset(Dataset):
    """Dataset that reads images from `train_images` and labels from `train.csv`."""

    def __init__(self, root_dir: str, csv_file: Union[str, pd.DataFrame], img_size=(384, 384), is_train=True):
        self.root_dir = Path(root_dir)
        if isinstance(csv_file, pd.DataFrame):
            self.df = csv_file
        else:
            self.df = pd.read_csv(csv_file)
        self.img_size = img_size
        self.is_train = is_train

        if is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=img_size),
                A.CoarseDropout(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
        else:
            self.transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / f"{row['image_id']}"
        image = np.array(Image.open(img_path).convert("RGB"))  # ✅ 轉為 NumPy 格式
        augmented = self.transform(image=image)                # ✅ 餵給 Albumentations
        image = augmented['image']                             # ✅ 取回 tensor
        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label


def get_loaders(data_dir: str, csv_path: str, batch_size: int = 16, val_ratio: float = 0.1, img_size: int = 384):
    full_df = pd.read_csv(csv_path)
    val_len = int(len(full_df) * val_ratio)
    train_len = len(full_df) - val_len

    indices = np.random.permutation(len(full_df))
    # train_indices = indices[:train_len]
    train_indices = indices 
    val_indices = indices[train_len:]

    train_df = full_df.iloc[train_indices].reset_index(drop=True)
    val_df = full_df.iloc[val_indices].reset_index(drop=True)

    train_dataset = CassavaDataset(data_dir, train_df, is_train=True, img_size=(img_size, img_size))
    val_dataset = CassavaDataset(data_dir, val_df, is_train=False, img_size=(img_size, img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader




if __name__ == "__main__":
    tl, vl = get_loaders("./cassava-leaf-disease-classification/train_images/",
                         "./cassava-leaf-disease-classification/train.csv", batch_size=4)
    images, labels = next(iter(tl))
    print(images.shape, labels.shape)
