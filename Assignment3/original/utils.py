# --- std / third‑party imports ------------------------------------------------
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torchvision.ops import masks_to_boxes
from pycocotools import mask as mask_utils
import skimage.io as sio

# -----------------------------------------------------------------------------
# COCO mask helpers
# -----------------------------------------------------------------------------

def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)


def encode_mask(mask_bin):
    rle = mask_utils.encode(np.asfortranarray(mask_bin))
    rle["counts"] = rle["counts"].decode("utf-8")         # ✅ 與 infer.py 一致
    return rle


def read_maskfile(filepath):
    return sio.imread(filepath)

# -----------------------------------------------------------------------------
# Albumentations transform factories
# -----------------------------------------------------------------------------

'''
alpha:形變幅度
sigma:平滑程度
'''
def get_train_transform() -> A.Compose:
    return A.Compose([
        # A.HorizontalFlip(p=0.3),
        # A.VerticalFlip(p=0.3),
        A.ElasticTransform(alpha=30, sigma=12, p=0.3),
        # A.RandomRotate90(p=0.3),
        # A.Normalize(mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# -----------------------------------------------------------------------------
# Dataset definitions
# -----------------------------------------------------------------------------

class CellDataset(Dataset):
    """Dataset for training / validation (with ground‑truth)."""

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.samples = sorted(self.root.iterdir())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        folder = self.samples[idx]
        img = np.array(Image.open(folder / "image.tif").convert("RGB"))
        H, W, _ = img.shape

        masks = []
        boxes = []
        labels = []

        for cls in range(1, 5):
            m_path = folder / f"class{cls}.tif"
            if not m_path.exists():
                continue
            mask = read_maskfile(m_path)
            inst_num = int(mask.max())

            for inst_id in range(1, inst_num + 1):
                inst_mask = (mask == inst_id).astype(np.uint8)
                pos = np.where(inst_mask > 0)
                if pos[0].size == 0 or pos[1].size == 0:
                    continue
                x_min, y_min = pos[1].min(), pos[0].min()
                x_max, y_max = pos[1].max(), pos[0].max()
                boxes.append([x_min, y_min, x_max, y_max])
                masks.append(inst_mask)
                labels.append(cls)

        # --- 3. 若沒有任何物件，補上空 target（模型不會爆） ---
        if len(masks) == 0:
            masks = torch.zeros((0, H, W), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

            # 對 image 做 basic Tensor 轉換（不使用 Albumentations）
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        else:
            # ✅ 清洗 masks 確保符合 Albumentations 要求
            masks_np = [np.array(m, dtype=np.uint8) for m in masks]

            if self.transform is not None:
                augmented = self.transform(image=img, masks=masks_np)
                img = augmented["image"]
                masks = augmented["masks"]
            else:
                img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }


        return img, target


class TestDataset(Dataset):
    """Dataset for submission inference (no ground‑truth)."""

    def __init__(self, root: str,
                 mapping_file: str = "./data/test_image_name_to_ids.json",
                 transform: A.Compose | None = None):
        self.root = Path(root)
        self.images = sorted(self.root.glob("*.tif"))
        with open(mapping_file, "r") as f:
            mapping = {item["file_name"]: item["id"] for item in json.load(f)}
        self.image_id_map = mapping
        self.transform = transform or get_val_transform()
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]      # Tensor [C,H,W]
        img_id = self.image_id_map[img_path.name]         # 👉 純 int
        return img, img_id


# -----------------------------------------------------------------------------
# DataLoader helpers
# -----------------------------------------------------------------------------

def collate_fn(batch: list[Tuple[torch.Tensor, dict]]):
    return tuple(zip(*batch))


def _split_train_val(root="data/train", val_ratio: float = 0.2, seed: int = 42):
    ds = CellDataset(root, transform=get_train_transform())
    val_sz = int(len(ds) * val_ratio)
    tr_sz = len(ds) - val_sz
    generator = torch.Generator().manual_seed(seed)
    return random_split(ds, [tr_sz, val_sz], generator=generator)


def get_train_loader(batch_size=2, num_workers=4, root="data/train"):
    train_ds = CellDataset(root, transform=get_train_transform())
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=True)


def get_val_loader(batch_size=2, num_workers=4, root="data/train"):
    _, val_ds = _split_train_val(root)
    val_ds.dataset.transform = get_val_transform()
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=True)


def get_test_loader(batch_size=2, num_workers=4, root="data/test_release"):
    test_ds = TestDataset(root)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=True)

# -----------------------------------------------------------------------------
# Quick sanity test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("[utils_torchvision] Sanity check …")

    dummy = np.zeros((32, 32), np.uint8)
    dummy[8:24, 8:24] = 1
    rle = encode_mask(dummy)
    assert np.array_equal(decode_maskobj(rle), dummy), "RLE round‑trip failed"
    print("  RLE ✔")

    if Path("./data/train").exists():
        loader = get_train_loader(batch_size=1, num_workers=0)
        imgs, targets = next(iter(loader))
        print(f"  Train batch: img {imgs[0].shape}, "
              f"masks {targets[0]['masks'].shape}, "
              f"boxes {targets[0]['boxes'].shape}, "
              f"labels {targets[0]['labels']}")
    
    if Path("./data/test_release").exists():
        loader = get_test_loader(batch_size=1, num_workers=0)
        imgs, ids = next(iter(loader))
        print(f"  Test batch: img {imgs[0].shape}, id {ids[0]}")
    
    print("All checks passed.")

