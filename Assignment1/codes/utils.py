import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import cv2

batch_size = 32

def set_batch(batch):
    batch_size = batch

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.2), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name


train_dataset = datasets.ImageFolder(root='./data/train', transform=train_transforms)


sorted_classes = sorted(train_dataset.class_to_idx.keys(), key=int)
new_class_to_idx = {cls: i for i, cls in enumerate(sorted_classes)}

train_dataset.class_to_idx = new_class_to_idx

train_dataset.samples = [(path, new_class_to_idx[os.path.basename(os.path.dirname(path))]) for path, _ in train_dataset.samples]
train_dataset.targets = [new_class_to_idx[os.path.basename(os.path.dirname(path))] for path, _ in train_dataset.imgs]

val_dataset = datasets.ImageFolder(root='./data/val', transform=val_transforms)

val_dataset.class_to_idx = new_class_to_idx
val_dataset.samples = [(path, new_class_to_idx[os.path.basename(os.path.dirname(path))]) for path, _ in val_dataset.samples]
val_dataset.targets = [new_class_to_idx[os.path.basename(os.path.dirname(path))] for path, _ in val_dataset.imgs]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

test_dir = "./data/test"
test_dataset = TestDataset(test_dir, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def get_train_loader(batch):
    set_batch(batch)
    return train_loader

def get_val_loader(batch):
    set_batch(batch)
    return val_loader

def get_test_loader(batch):
    set_batch(batch)
    return test_loader

# dynamic train data augmentation
def mixup_data(x, y, alpha=1.0):
    '''MixUp augmentation'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)  # mix up index

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''MixUp Loss 計算'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    '''CutMix augmentation'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    W, H = x.size(2), x.size(3)
    r_x, r_y = np.random.randint(W), np.random.randint(H)
    r_w, r_h = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))

    x1, y1 = max(0, r_x - r_w // 2), max(0, r_y - r_h // 2)
    x2, y2 = min(W, r_x + r_w // 2), min(H, r_y + r_h // 2)

    x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))  # 更新 lambda
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    if train_dataset.class_to_idx == val_dataset.class_to_idx:
        print(" Train 和 Validation 的類別索引匹配！")
    else:
        print("類別索引不匹配！請檢查 class_to_idx 設定")

    # 測試讀取一批訓練數據
    try:
        train_images, train_labels = next(iter(train_loader))
        print(f"Train batch shape: {train_images.shape}, Labels shape: {train_labels.shape}")
    except Exception as e:
        print(f"Error loading train data: {e}")

    # 測試讀取一批驗證數據
    try:
        val_images, val_labels = next(iter(val_loader))
        print(f"Validation batch shape: {val_images.shape}, Labels shape: {val_labels.shape}")
    except Exception as e:
        print(f"Error loading validation data: {e}")

    # 測試讀取一批測試數據
    try:
        test_images, test_filenames = next(iter(test_loader))
        print(f"Test batch shape: {test_images.shape}")
        print(f"Sample filenames: {test_filenames[:5]}")  # 列出前5個檔名
    except Exception as e:
        print(f"Error loading test data: {e}")

