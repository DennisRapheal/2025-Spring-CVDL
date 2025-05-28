# ------------------------------------------------------------
# train.py  –  Cassava Leaf Disease Classification (PyTorch)
# ------------------------------------------------------------
import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import GradScaler, autocast

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

# local imports
from model import HybridCassavaNet, EfficientNetClassifier
from utils import get_loaders


# --------------------------- utils ---------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(1)
    return (preds == targets).float().mean().item()


def log_row(csv_file: Path, row: Dict) -> None:
    write_header = not csv_file.exists()
    with csv_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# --------------------------- train / val ---------------------------
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    use_amp=False,
):
    model.train()
    running_loss, running_acc = 0.0, 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_acc += accuracy(outputs, labels) * imgs.size(0)

        pbar.set_postfix(loss=loss.item())

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss, running_acc = 0.0, 0.0

    pbar = tqdm(loader, desc=f"Val   Epoch {epoch}", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        running_acc += accuracy(outputs, labels) * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


# --------------------------- main ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="./cassava-leaf-disease-classification/train_images/")
    p.add_argument("--csv_path", type=str,
                   default="./cassava-leaf-disease-classification/train.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--name", type=str, default="best_model")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--eff_only", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true",
                   help="enable mixed-precision training")
    p.add_argument("--no_pretrained", action="store_true",
                   help="dont load EfficientNet pretrained weights")
    p.add_argument("--outdir", type=str, default="./outputs")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.name:
        outdir = outdir / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    log_csv = outdir / "train_log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    train_loader, val_loader = get_loaders(
        args.data_dir,
        args.csv_path,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        img_size=args.img_size,
    )

    # model
    if args.eff_only:
        model = EfficientNetClassifier(
            num_classes=5,
            pretrained=not args.no_pretrained,
        ).to(device)
    else:
        model = HybridCassavaNet(
            num_classes=5,
            pretrained_eff=not args.no_pretrained,
        ).to(device)

    model = torch.nn.DataParallel(model)

    # loss / optim / sched
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,   # 第一次重啟前的 iteration 數
        T_mult=2,            # 之後週期倍增
        eta_min=0.0              # 不使用 m_mul 時，讓 LR 下降到 0
    )

    scaler = GradScaler(enabled=args.amp)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, use_amp=args.amp)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch)

        scheduler.step()

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # log csv
        log_row(log_csv, {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": scheduler.get_last_lr()[0],
        })

        # checkpoint
        if args.val_ratio == 0.0 or val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = outdir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "acc": best_acc,
                },
                ckpt_path,
            )
            print(f"✅  Saved new best model to {ckpt_path} (acc={best_acc:.4f})")

    print("Training finished.")


if __name__ == "__main__":
    main()
