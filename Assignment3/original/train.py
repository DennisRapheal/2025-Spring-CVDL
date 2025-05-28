# train.py – instance‑segmentation training loop
# -------------------------------------------------
#  Usage:
#      python train.py --epochs 50 --batch_size 4 --backbone resnet50

import os
import time
import argparse
from pathlib import Path
import pandas as pd

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from utils import get_train_loader, get_val_loader
from model import get_model

# -------------------------
# CLI args
# -------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='unknonw')
    p.add_argument('--model_type', type=str, default='resnet50',
                   choices=['resnet50', 'resnet50_v2'])
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--data_root', type=str, default='data/train')
    p.add_argument('--save_dir', type=str, default='checkpoints')
    return p.parse_args()



# -------------------------
# Epoch loops
# -------------------------

def train_one_epoch(model, loader, optim, device):
    model.train()
    metric = MeanAveragePrecision(iou_type="bbox")  # 先用 bbox mAP 作 proxy

    epoch_loss = 0.0

    for images, targets in loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        loss = sum(loss_dict.values())
        epoch_loss += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        # mAP (bbox)
        model.eval()
        with torch.no_grad():
            preds = model(images)
        model.train()

        preds_cpu = [{
            'boxes': p['boxes'].cpu(),
            'scores': p['scores'].cpu(),
            'labels': p['labels'].cpu()
        } for p in preds]
        gts_cpu = [{
            'boxes': t['boxes'].cpu(),
            'labels': t['labels'].cpu()
        } for t in targets]
        metric.update(preds_cpu, gts_cpu)

    stats = metric.compute()
    return epoch_loss / len(loader), stats['map'].item(), stats['map_50'].item()


def validate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(images)
            preds_cpu = [{
                'boxes': p['boxes'].cpu(),
                'scores': p['scores'].cpu(),
                'labels': p['labels'].cpu()
            } for p in preds]
            gts_cpu = [{
                'boxes': t['boxes'].cpu(),
                'labels': t['labels'].cpu()
            } for t in targets]
            metric.update(preds_cpu, gts_cpu)
    stats = metric.compute()
    return stats['map'].item(), stats['map_50'].item()

# -------------------------
# Main
# -------------------------

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader = get_train_loader(batch_size=args.batch_size, root=args.data_root)
    val_loader = get_val_loader(batch_size=args.batch_size, root=args.data_root)

    # Model
    model = get_model(num_classes=5, model_type=args.model_type)
    model.to(device)

    # Optimiser & scheduler
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs)

    # Logging / ckpt dir
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    best_score = 0.0
    history = []
    print(f"Start training on {device}!  Epochs={args.epochs}  Batch={args.batch_size}")

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_map, train_map50 = train_one_epoch(
            model, train_loader, optimiser, device)
        val_map, val_map50 = validate(model, val_loader, device)
        scheduler.step()
        epoch_time = time.time() - start

        history.append([
            epoch + 1, train_loss, train_map, train_map50, val_map, val_map50
        ])

        print(f"E{epoch+1:03d}/{args.epochs} | "
              f"loss {train_loss:.4f} | "
              f"train mAP {train_map:.4f}/{train_map50:.4f} | "
              f"val mAP {val_map:.4f}/{val_map50:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | "
              f"{epoch_time/60:.1f} min")

        map_sum = val_map + val_map50
        if map_sum >= best_score:
            best_score = map_sum
            ckpt = Path(args.save_dir) / f"{args.model_name}_maskrcnn.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  ↳ New best! saved to {ckpt}")

    print("Training finished. Best val mAP: ", best_score)
    # 儲存訓練歷史
    os.makedirs(f'./TrainingLog', exist_ok=True)
    df = pd.DataFrame(history, columns=[
        'Epoch', 'Train Loss', 'Train mAP', 'Train mAP@0.5',
        'Val mAP', 'Val mAP@0.5'
    ])
    df.to_csv(f'./TrainingLog/{args.model_name}.csv', index=False)
    print(f"Training log saved to ./TrainingLog/{args.model_name}.csv")
