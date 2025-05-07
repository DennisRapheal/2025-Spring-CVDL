import os
import torch
import argparse
import pandas as pd
import time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import get_train_loader, get_easy_train_loader, get_val_loader, set_batch
from model import get_model
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# -------------------------
# Args
# -------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")
    parser.add_argument("--model_name", type=str, default="resnext50_32x4d")
    parser.add_argument("--model_type", type=str, default="resnext50_32x4d")
    parser.add_argument("--from_pretrained", type=str, default="")
    parser.add_argument("--transform_type", type=str, default="easy")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--schedulerT_0", type=int, default=200)
    parser.add_argument("--schedulerT_mult", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


# -------------------------
# Train One Epoch
# -------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs, scheduler):
    model.train()
    metric = MeanAveragePrecision(iou_type="bbox")
    total_loss = 0.0
    total_batches = len(loader)
    torch.cuda.empty_cache()

    for i, (images, targets) in enumerate(loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 過濾掉無 bbox 的樣本
        filtered = [
            (img, tgt) for img, tgt in zip(images, targets)
            if tgt["boxes"].numel() > 0 and tgt["boxes"].shape[1] == 4
        ]
        if len(filtered) == 0:
            continue  # 如果這個 batch 全部都沒有 bbox，就跳過
        images, targets = zip(*filtered)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        # --- Compute IoU & mAP ---
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        model.train()

        preds = []
        gts = []
        for output in outputs:
            preds.append({
                "boxes": output["boxes"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "labels": output["labels"].detach().cpu()
            })
        for target in targets:
            gts.append({
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu()
            })

        metric.update(preds, gts)

    avg_loss = total_loss / total_batches
    result = metric.compute()
    ap = result["map"].item()
    ap50 = result["map_50"].item()

    print(f"[Train] Epoch {epoch+1}/{total_epochs} Summary:")
    print(f"Avg Loss: {avg_loss:.4f} | mAP@[.5:.95]: {ap:.4f} | mAP@0.5: {ap50:.4f}")

    return avg_loss, ap, ap50



# -------------------------
# Val One Epoch (simple acc proxy based on IoU match)
# -------------------------
def val_one_epoch(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    correct_cls = 0

    with torch.no_grad():
        for images, targets in loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            preds = []
            gts = []
            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu()
                })
            for target in targets:
                gts.append({
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu()
                })

            metric.update(preds, gts)

    result = metric.compute()
    ap = result["map"].item()
    ap50 = result["map_50"].item()

    print(f"[Val] mAP@[.5:.95]: {ap:.4f} | mAP@0.5: {ap50:.4f}")
    return ap, ap50


# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    args = get_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set batch size
    set_batch(args.batch_size)

    # Load data
    if args.transform_type == 'easy':
        train_loader = get_easy_train_loader()
    else: 
        train_loader = get_train_loader()

    val_loader = get_val_loader()

    # Load model
    if args.from_pretrained == "":
        model = get_model(args.model_type)
    else:
        model = get_model(model_type=args.model_type, weights_pth=args.from_pretrained)
    
    model.to(device)

    # Optimizer / Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Checkpoint saving
    save_path = './checkpoints'
    os.makedirs(save_path, exist_ok=True)

    # History
    history = []

    print("Start Training!")
    print(f"Using device: {device} | GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    max_map_sum = 0.0
    
    for epoch in range(args.num_epochs):
        train_loss, train_map, train_map50 = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.num_epochs, scheduler
        )
        val_map, val_map50 = val_one_epoch(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]

        history.append([
            epoch + 1, train_loss, train_map, train_map50, val_map, val_map50, lr_now
        ])
        scheduler.step()

        cur_map_sum = val_map + train_map

        if cur_map_sum >= max_map_sum:
            max_map_sum = cur_map_sum
            ckpt_path = os.path.join(save_path, f"{args.model_name}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f" Saved checkpoint: {ckpt_path}")

    # 儲存訓練歷史
    os.makedirs(f'./weights/{args.model_name}', exist_ok=True)
    df = pd.DataFrame(history, columns=[
        'Epoch', 'Train Loss', 'Train mAP', 'Train mAP@0.5',
        'Val mAP', 'Val mAP@0.5', 'LR'
    ])
    df.to_csv(f'./weights/{args.model_name}/training_log.csv', index=False)
    print(f"Training log saved to ./weights/{args.model_name}/training_log.csv")
