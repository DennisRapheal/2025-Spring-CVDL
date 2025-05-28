# ---------------------------------------------------------------
# inference.py  –  PromptIR blind restoration *inference‑only* script
# ---------------------------------------------------------------
"""
用法範例
---------
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --ckpt ckpt/epoch=149-step=9999.ckpt \
  --input_dir ./data/test/degraded \
  --output_dir ./runs/predict

*   `ckpt`：train.py 由 ModelCheckpoint 產出的 .ckpt
*   `input_dir`：資料夾中全部都是 degraded/unknown 圖片 (png/jpg/…)
*   `output_dir`：推論後圖片輸出位置，若不存在會自動建立
"""

import argparse, os, glob, math, cv2, torch
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

import numpy as np

# repo imports ---------------------------------------------------
from net.model import PromptIR                             # 你的模型 backbone

# ---------------------------------------------------------------
# Dataset: 讀資料夾所有影像，不需要 GT
# ---------------------------------------------------------------

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

def scan_dir(root):
    """recursively collect valid image paths, keep order deterministic"""
    files = []
    for ext in EXT:
        files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted(files)

class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.paths = scan_dir(root)
        if not self.paths:
            raise RuntimeError(f"[inference] No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = cv2.imread(path)[:, :, ::-1]                # BGR -> RGB
        tensor = to_tensor(img / 255.)                     # HWC [0‑1] -> CHW float32
        return os.path.basename(path), tensor

# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------

def load_lightning_ckpt(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    net = PromptIR(decoder=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint
    net.load_state_dict(state_dict, strict=True)
    net.to(device).eval()
    return net


def parse_args():
    p = argparse.ArgumentParser(description="PromptIR blind restoration inference")
    p.add_argument("--ckpt", required=True, help="Path to .ckpt from training")
    p.add_argument("--input_dir", required=True, help="Folder with degraded images")
    p.add_argument("--output_dir", default="output_pred", help="Where to save restored images")
    p.add_argument("--batch_size", type=int, default=4, help=">1 會自動組 batch (適合相同大小影像)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--half", action="store_true", help="Use torch.float16 during inference")
    p.add_argument("--cuda", type=int, default=0, help="CUDA device idx (‑1 使用 CPU)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu")

    # 1. 模型 -----------------------------------------------------
    print(f"[inference] Loading model from {args.ckpt} …")
    net = load_lightning_ckpt(args.ckpt, device)

    # 2. DataLoader ----------------------------------------------
    ds = FolderDataset(args.input_dir)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 推論 ------------------------------------------------------
    dtype = torch.float16 if args.half and device.type == "cuda" else torch.float32

    with torch.no_grad():
        for names, batch in tqdm(dl, total=len(dl), unit="img"):
            batch = batch.to(device, dtype=dtype)
            out   = net(batch).clamp(0, 1)

            # save each image in batch
            img_folder = os.path.join(args.output_dir, "pred_images")
            os.makedirs(img_folder, exist_ok=True)

            for i, name in enumerate(names):
                save_path = os.path.join(img_folder, name)
                save_image(out[i], save_path)  # 儲存為 RGB 圖片，自動 *255 處理

    print(f"[inference] Done. Restored images are in {args.output_dir}")
    
    # 4. 產生 pred.npz 檔案 --------------------------------------
    print(f"[inference] Saving pred.npz and pred.zip ...")
    images_dict = {}
    for filename in os.listdir(img_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            file_path = os.path.join(img_folder, filename)
            image = cv2.imread(file_path)  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
            img_array = np.transpose(image, (2, 0, 1))  # HWC → CHW
            images_dict[filename] = img_array.astype(np.uint8)

    npz_path = os.path.join(args.output_dir, 'pred.npz')
    np.savez(npz_path, **images_dict)

    import zipfile
    zip_path = os.path.join(args.output_dir, 'pred.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(npz_path, arcname='pred.npz')

    print(f"[inference] Done. Saved all .png to {img_folder}, and pred.npz/zip to {args.output_dir}")


if __name__ == "__main__":
    main()
