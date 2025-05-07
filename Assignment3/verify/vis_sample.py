import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from PIL import Image
from pathlib import Path
import skimage.io as sio

# ==== 設定路徑與目標 ====
pred_json_path = "./../sample-image_result/pred.json"
test_img_dir = Path("./../sample-image")
test_cls_dir = Path("./../sample-image-cls")
target_id = 3

# ==== 類別顏色對應表（category_id → RGB） ====
CLASS_COLORS = {
    1: (255, 0, 0),    # Red
    2: (0, 255, 0),    # Green
    3: (255, 255, 0),    # Blue
    4: (0, 0, 255),  # Yellow
}
alpha = 0.5  # 半透明度

# ==== 讀取 prediction.json ====
with open(pred_json_path, "r") as f:
    preds = json.load(f)

# ==== 篩選對應 image_id 的預測 ====
img_preds = [p for p in preds if p["image_id"] == target_id]
if not img_preds:
    print(f"[WARN] No prediction found for image_id={target_id}")
    exit()

# ==== 讀取原圖 ====
img_path = test_img_dir / "acaf16a8-d1e6-4746-92ae-c76fa82cb70d.tif"
if not img_path.exists():
    raise FileNotFoundError(f"{img_path} not found.")
orig_img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)

img_h, img_w = orig_img.shape[:2]

# ==== 疊上 Ground Truth 遮罩（使用 class1～class4） ====
gt_overlay = orig_img.copy()
for cls in range(1, 5):
    mask_path = test_cls_dir / f"class{cls}.tif"
    if mask_path.exists():
        mask = sio.imread(str(mask_path))
        mask_bin = (mask > 0).astype(np.uint8)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        for i in range(3):
            gt_overlay[..., i] = np.where(
                mask_bin,
                gt_overlay[..., i] * (1 - alpha) + color[i] * alpha,
                gt_overlay[..., i]
            )

# ==== 疊上預測遮罩 ====
pred_overlay = orig_img.copy()
for p in img_preds:
    mask = mask_utils.decode(p["segmentation"])
    category_id = p.get("category_id", 0)
    color = CLASS_COLORS.get(category_id, (255, 255, 255))
    for i in range(3):
        pred_overlay[..., i] = np.where(
            mask,
            pred_overlay[..., i] * (1 - alpha) + color[i] * alpha,
            pred_overlay[..., i]
        )

# ==== 顯示兩張圖 ====
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(gt_overlay.astype(np.uint8))
plt.title("Ground Truth Overlay")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred_overlay.astype(np.uint8))
plt.title("Prediction Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()
