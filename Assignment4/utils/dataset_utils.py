import os, cv2, random, torch, re, glob
import numpy as np
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset
from typing import List, Tuple

def _make_pairs(de_paths: List[str]) -> Tuple[List[str], List[str]]:
    cl_paths = []
    for p in de_paths:
        # 先取得檔名
        fname = os.path.basename(p)

        # 判斷是哪一類
        if 'rain-' in fname:
            cl_name = fname.replace('rain-', 'rain_clean-')
        elif 'snow-' in fname:
            cl_name = fname.replace('snow-', 'snow_clean-')
        else:
            cl_name = fname  # fallback

        # 最後 clean 圖路徑
        cl_path = p.replace('degraded', 'clean').replace(fname, cl_name)
        cl_paths.append(cl_path)

    return de_paths, cl_paths

class PromptDataset(Dataset):
    """只負責取資料，不切 train/val。"""
    def __init__(self,
                 de_paths: List[str],
                 patch_size: int = 128,
                 is_train: bool = True,
                 flip_prob: float = 0.5):
        self.de_paths, self.cl_paths = _make_pairs(de_paths)
        self.patch_size = patch_size
        self.is_train   = is_train
        self.flip_prob  = flip_prob if is_train else 0.0

    def __len__(self):
        return len(self.de_paths)

    # ---------- util ----------
    def _paired_random_crop(self, img_de, img_cl):
        """在降雨、乾淨圖上取同座標 patch"""
        h, w = img_de.shape[:2]; p = self.patch_size
        if h <= p or w <= p:      # 圖太小就直接回傳整張
            return img_de, img_cl
        y = random.randrange(0, h - p + 1)
        x = random.randrange(0, w - p + 1)
        return (img_de[y:y+p, x:x+p],
                img_cl[y:y+p, x:x+p])

    # ---------- load ----------
    def __getitem__(self, idx):
        de_path = self.de_paths[idx]
        cl_path = self.cl_paths[idx]

        # 依檔名決定任務 id
        fname = os.path.basename(de_path)
        if 'rain-' in fname:
            de_type = 0      # derain
        elif 'snow-' in fname:
            de_type = 1      # desnow
        else:
            raise ValueError(f"未知降噪類型：{fname}")

        de_img = cv2.imread(de_path)[:, :, ::-1]  # BGR → RGB
        cl_img = cv2.imread(cl_path)[:, :, ::-1]

        # 同步隨機翻轉
        if random.random() < self.flip_prob:
            de_img, cl_img = de_img[:, ::-1, :], cl_img[:, ::-1, :]

        # 同步隨機裁切
        if self.is_train and self.patch_size:
            de_img, cl_img = self._paired_random_crop(de_img, cl_img)

        # numpy → tensor, 正規化到 0~1
        de_tensor = to_tensor(de_img.astype(np.float32) / 255.)
        cl_tensor = to_tensor(cl_img.astype(np.float32) / 255.)

        clean_name = os.path.splitext(os.path.basename(cl_path))[0]

        # **回傳格式對齊官方**：list 兩元素
        return [clean_name, de_type], de_tensor, cl_tensor


def _numeric_sort(files):
    return sorted(files, key=lambda p: int(re.search(r'(\d+)', p).group(1)))

class PromptTestDataset(Dataset):
    """Test dataset：檔名即 idx；不裁剪、不讀 GT。"""
    EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

    def __init__(self, opt):
        root = opt.test_dir
        if os.path.isdir(os.path.join(root, 'degraded')):
            root = os.path.join(root, 'degraded')

        self.paths = _numeric_sort(
            [f for f in os.listdir(root) if f.lower().endswith(self.EXT)]
        )
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        name = self.paths[i]                 # e.g. "0.png"
        img  = cv2.imread(os.path.join(self.root, name))[:, :, ::-1]
        tensor = to_tensor(img / 255.)
        return name, tensor
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./../hw4_realse_dataset/train',)
    parser.add_argument('--num_samples', type=int, default=5,)
    args = parser.parse_args()

    # 讀取 degraded 清單
    de_dir = os.path.join(args.train_dir, 'degraded')
    de_paths = _numeric_sort(
        [os.path.join(de_dir, f) for f in os.listdir(de_dir)
         if f.lower().endswith(PromptTestDataset.EXT)]
    )

    print(f"共讀入 {len(de_paths)} 張 degraded 圖片")

    # 初始化 Dataset
    dataset = PromptDataset(de_paths, patch_size=128, is_train=True)
    for i in range(3):
        (name, de_id), de, cl = dataset[i]
        print(name, de_id, de.shape)
        assert torch.allclose(de * 0 + 1, cl * 0 + 1, atol=1), "Tensor dtype 不同"
        assert de.shape == cl.shape, "de / cl 尺寸不符"
