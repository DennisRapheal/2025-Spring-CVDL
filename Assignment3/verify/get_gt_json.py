import os
import numpy as np
from pathlib import Path
from PIL import Image
from pycocotools import mask as mask_utils
import skimage.io as sio
import json

# ==== 設定 ====
image_id = 3
image_file = "acaf16a8-d1e6-4746-92ae-c76fa82cb70d.tif"
image_dir = Path("./../sample-image")
cls_dir =  Path("./../sample-image-cls")

class_ids = [1, 2, 3, 4]  # class1.tif ~ class4.tif
gt_json_path = "gt.json"

# ==== 建立 COCO 資料結構 ====
coco_gt = {
    "images": [{
        "id": image_id,
        "width": None,  # 稍後填入
        "height": None,
        "file_name": image_file
    }],
    "annotations": [],
    "categories": [
        {"id": cid, "name": f"class{cid}"} for cid in class_ids
    ]
}

ann_id = 1  # annotation ID 遞增用

# ==== 讀入原圖大小 ====
img_path = image_dir / image_file
img = np.array(Image.open(img_path))
height, width = img.shape[:2]
coco_gt["images"][0]["width"] = width
coco_gt["images"][0]["height"] = height

# ==== 處理每個 class mask ====
for cid in class_ids:
    mask_path = cls_dir / f"class{cid}.tif"
    if not mask_path.exists():
        print(f"[WARN] Mask {mask_path.name} not found, skip.")
        continue

    mask = sio.imread(mask_path)
    binary_mask = (mask > 0).astype(np.uint8)
    if binary_mask.sum() == 0:
        continue

    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")  # JSON 不接受 bytes

    ann = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": cid,
        "segmentation": rle,
        "area": float(mask_utils.area(rle)),
        "bbox": list(mask_utils.toBbox(rle)),
        "iscrowd": 0
    }
    coco_gt["annotations"].append(ann)
    ann_id += 1

# ==== 寫入 gt.json ====
with open(gt_json_path, "w") as f:
    json.dump(coco_gt, f, indent=2)

print(f"[INFO] Saved GT to {gt_json_path}")
