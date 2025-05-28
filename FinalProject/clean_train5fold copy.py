# cassava_training_clean.py
"""
Cassava Leaf Disease Classification – Clean Training Script
==========================================================
This script trains a ConvNeXt‑based classifier on the Cassava Leaf Disease
Classification dataset using 5‑fold cross‑validation.  Major features:

* Albumentations for data augmentation (mix‑up & cut‑mix supported via timm)
* GeM global pooling + lightweight classification head
* Cosine or plateau LR scheduling
* Mixed logging to console & file via Python logging

All core sections are grouped and fully documented so you can quickly adapt
hyper‑parameters or swap the backbone.
"""

# --------------------------------------------------
# 1. Imports & Environment
# --------------------------------------------------
from __future__ import annotations

import os
import math
import random
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import (
    Compose, RandomResizedCrop, HorizontalFlip, VerticalFlip, ShiftScaleRotate,
    CoarseDropout, Normalize, Resize, Transpose
)
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import timm

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 2. Global configuration – tune here
# --------------------------------------------------
class CFG:
    # experiment
    seed: int = 42
    num_epochs: int = 20
    n_folds: int = 5
    folds_for_training: List[int] = [0, 1, 2, 3, 4]

    # data
    img_size: int = 512
    batch_size: int = 32
    num_workers: int = 2  # set 0 on Windows

    # model
    backbone: str = "tf_efficientnetv2_l_in21k"
    pretrained: bool = True
    num_classes: int = 4
    drop_path_rate: float = 0.2

    # optimisation
    lr: float = 1e-4
    weight_decay: float = 1e-3
    max_grad_norm: float = 1_000.0
    scheduler: str = "cosine"          # "cosine" | "plateau"
    t_max: int = 20                     # cosine
    min_lr: float = 5e-7
    factor: float = 0.2                 # plateau
    patience: int = 5                   # plateau
    eps: float = 1e-6                   # plateau

    # augmentation
    mixup_prob: float = 0.5             # prob 0 → disable mix & cut‑mix
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1

    # logging / misc
    print_freq: int = 100
    output_dir: Path = Path("./outputs")
    train_path: Path = Path("cassava-leaf-disease-classification/train_images")
    test_path: Path = Path("cassava-leaf-disease-classification/test_images")


CFG.output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3. Reproducibility helper
# --------------------------------------------------
import logging
log_path = CFG.output_dir / "train.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(CFG.seed)

# --------------------------------------------------
# 4. Small utility classes & functions
# --------------------------------------------------
class AverageMeter:
    """Stores and updates a series of values (e.g. loss over an epoch)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy helper."""
    return accuracy_score(y_true, y_pred)


def as_minutes(s: float) -> str:
    m = math.floor(s / 60)
    return f"{m}m {int(s - m * 60)}s"


def time_since(start: float, progress: float) -> str:
    """Return elapsed/remaining time string."""
    now = time.time()
    elapsed = now - start
    total = elapsed / progress
    remain = total - elapsed
    return f"{as_minutes(elapsed)} (remain {as_minutes(remain)})"

# --------------------------------------------------
# 5. Dataset & Augmentations
# --------------------------------------------------
class CassavaDataset(Dataset):
    """A simple (image, label) dataset."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["image_id"].values
        self.labels = df["label"].values.astype(int)
        self.transform = transform

    def __len__(self) -> int:  # noqa: D401
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = CFG.train_path / self.paths[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        
        # only train 4 label
        label = self.labels[idx]
        if label == 4:
            label = 3  # ✅ 把 4 映射為 class index 3
        label = torch.tensor(label).long()
        return image, label


def get_transforms(stage: str):
    """Albumentations pipeline for *stage* ("train"|"valid")."""
    if stage == "train":
        return Compose([
            RandomResizedCrop((CFG.img_size, CFG.img_size)),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            CoarseDropout(max_holes=8, max_height=CFG.img_size // 8,
                          max_width=CFG.img_size // 8, p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:  # valid / test
        return Compose([
            Resize(CFG.img_size, CFG.img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# --------------------------------------------------
# 6. Model definition
# --------------------------------------------------
class GeM(nn.Module):
    """Generalised Mean Pooling (p is learnable)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        self.eps = eps
        self.p = nn.Parameter(torch.ones(1) * p) if trainable else torch.tensor([p])

    def forward(self, x):  # (B, C, H, W)
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)

    def __repr__(self):
        return f"GeM(p={self.p.data.item():.4f})"


class NetClassifier(nn.Module):
    """ConvNeXt backbone + GeM + simple FC head."""

    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, drop_path_rate=CFG.drop_path_rate,
            features_only=True, out_indices=[-1]
        )
        in_ch = self.backbone.feature_info.channels()[-1]
        self.pool = GeM()
        self.head = nn.Sequential(
            nn.Linear(in_ch, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)[0]
        x = self.pool(x).flatten(1)
        return self.head(x)

# --------------------------------------------------
# 7. Augmentation helpers (mix‑up & cut‑mix)
# --------------------------------------------------
mixup_fn = Mixup(
    mixup_alpha=CFG.mixup_alpha,
    cutmix_alpha=CFG.cutmix_alpha,
    prob=CFG.mixup_prob,
    switch_prob=0.5,
    mode="batch",
    label_smoothing=CFG.label_smoothing,
    num_classes=CFG.num_classes,
)
criterion_soft = SoftTargetCrossEntropy()
criterion_hard = nn.CrossEntropyLoss()

# --------------------------------------------------
# 8. Training / validation loops
# --------------------------------------------------

def train_one_epoch(loader, model, optimizer, scheduler, epoch):
    model.train()
    losses = AverageMeter()
    start = time.time()

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images, targets = mixup_fn(images, labels)
        preds = model(images)
        loss = criterion_soft(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        if step % CFG.print_freq == 0:
            print(
                f"Epoch [{epoch+1}] Step [{step}/{len(loader)}] "
                f"Loss {losses.val:.4f} (avg {losses.avg:.4f}) – "
                f"{time_since(start, step/len(loader)+1e-9)}"
            )
    scheduler.step()
    return losses.avg


def valid_one_epoch(loader, model):
    model.eval()
    losses = AverageMeter()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images)
            loss = criterion_hard(preds, labels)

            losses.update(loss.item(), images.size(0))
            all_preds.append(torch.softmax(preds, 1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = get_score(labels, preds.argmax(1))
    return losses.avg, acc, preds

# --------------------------------------------------
# 9. Training routine per fold
# --------------------------------------------------

def train_fold(folds_df: pd.DataFrame, fold: int) -> pd.DataFrame:
    print(f"\n========== Fold {fold} ==========")
    trn_idx = folds_df[folds_df["fold"] != fold].index
    val_idx = folds_df[folds_df["fold"] == fold].index

    # datasets / loaders
    train_ds = CassavaDataset(folds_df.loc[trn_idx].reset_index(drop=True),
                              transform=get_transforms("train"))
    valid_ds = CassavaDataset(folds_df.loc[val_idx].reset_index(drop=True),
                              transform=get_transforms("valid"))

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True)

    # model / optim
    model = NetClassifier(CFG.backbone, CFG.num_classes, CFG.pretrained)
    model = nn.DataParallel(model).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    if CFG.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.t_max, eta_min=CFG.min_lr
        )
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=CFG.factor, patience=CFG.patience, eps=CFG.eps
        )

    best_acc = 0.0
    oof_preds = np.zeros((len(valid_ds), CFG.num_classes), dtype=np.float32)

    for epoch in range(CFG.num_epochs):
        train_loss = train_one_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_loss, val_acc, preds = valid_one_epoch(valid_loader, model)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d} – TL {train_loss:.4f}  VL {val_loss:.4f}  ACC {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CFG.output_dir / f"best_fold{fold}.pth")
            oof_preds = preds  # keep best preds

    # attach predictions
    fold_df = folds_df.loc[val_idx].copy()
    fold_df[[str(i) for i in range(CFG.num_classes)]] = oof_preds
    fold_df["preds"] = oof_preds.argmax(1)
    return fold_df

# --------------------------------------------------
# 10. Main entry point – k‑fold CV
# --------------------------------------------------

def main():
    # training dataframe & externally prepared fold split file
    train_df = pd.read_csv("cassava-leaf-disease-classification/train.csv")
    folds_df = train_df.merge(
        pd.read_csv("validation_data.csv")[["image_id", "fold"]], on="image_id"
    )

    # 如果只想訓練 小的class
    folds_df = folds_df[folds_df['label'] != 3].reset_index(drop=True)

    oof_df = pd.DataFrame()
    for fold in range(CFG.n_folds):
        if fold in CFG.folds_for_training:
            fold_preds = train_fold(folds_df, fold)
            oof_df = pd.concat([oof_df, fold_preds])
            print(f"Fold {fold} ACC: {get_score(fold_preds['label'], fold_preds['preds']):.4f}")

    # CV summary
    cv_acc = get_score(oof_df['label'], oof_df['preds'])
    print(f"========== CV Accuracy: {cv_acc:.4f} ==========")
    oof_df.to_csv(CFG.output_dir / "oof_df.csv", index=False)


if __name__ == "__main__":
    main()
