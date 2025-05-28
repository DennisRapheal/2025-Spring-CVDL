import cv2
# ====================================================
# libraries
# ====================================================

import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from functools import partial
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from albumentations import (Compose, Normalize, Resize, RandomResizedCrop, HorizontalFlip, VerticalFlip, ShiftScaleRotate, Transpose)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import timm
import warnings 
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
from matplotlib import pyplot as plt

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = 'cassava-leaf-disease-classification/train_images'
TEST_PATH = 'cassava-leaf-disease-classification/test_images'

# ====================================================
# CFG
# ====================================================

class CFG:
    print_freq=100
    model_name = 'tf_efficientnetv2_xl'
    scheduler = 'cosine'
    T_max = 10
    mix_up = False
    mixup_prob = 0.7
    cut_mix = True
    cutmix_prob = 0.7
    num_workers = 0
    size = 512
    epochs = 30
    factor = 0.2
    patience = 5
    eps = 1e-6
    lr = 5e-5
    min_lr = 1e-6
    batch_size = 32
    weight_decay = 1e-2
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = 5
    target_col = 'label'
    n_fold = 5
    trn_fold = [0,1,2,3,4]

assert not (CFG.mix_up and CFG.cut_mix), "Cannot enable both MixUp and CutMix!"

# ====================================================
# utils
# ====================================================

def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')

def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

    logger = getLogger(__name__)  # 應該先取得 logger 實例

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)

# ====================================================
# dataset
# ====================================================

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TRAIN_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label

# ====================================================
# transformations
# ====================================================

def get_transforms(*, data):
    
    if data == 'train':
        return Compose([
            RandomResizedCrop((CFG.size, CFG.size)),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
       return Compose([
           Resize(CFG.size, CFG.size),
           Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
           ),
           ToTensorV2(),
       ])

# # ====================================================
# # model initialization
# # ====================================================


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()

        self.eff = timm.create_model(
            "tf_efficientnetv2_xl",
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],
        )
        eff_ch = self.eff.feature_info.channels()[-1]  # 1792

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(eff_ch, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        feats = self.eff(x)[0]  # (B, 1792, H/32, W/32)
        return self.classifier(feats)


# ====================================================
# helper functions
# ====================================================

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    model.train()
    start = end = time.time()
    global_step = 0

    for step, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # Augmentation switch: mixup or cutmix
        # --- Decide if mix or cutmix is applied ---
        use_mix = False

        if CFG.mix_up and np.random.rand() < CFG.mixup_prob:
            alpha = 0.4
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(batch_size).to(device)
            mixed_images = lam * images + (1 - lam) * images[index, :]
            labels_a, labels_b = labels, labels[index]
            inputs = mixed_images
            use_mix = True

        elif CFG.cut_mix and np.random.rand() < CFG.cutmix_prob:
            alpha = 1.0
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(batch_size).to(device)
            labels_a, labels_b = labels, labels[index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            inputs = images
            use_mix = True

        else:
            inputs = images
            labels_a = labels
            labels_b = labels
            lam = 1.0
            use_mix = False

        # --- Safe loss calculation ---
        y_preds = model(inputs)

        if use_mix:
            loss = lam * criterion(y_preds, labels_a) + (1 - lam) * criterion(y_preds, labels_b)
        else:
            loss = criterion(y_preds, labels_a)



        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   ))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.softmax(1).to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


# ====================================================
# train loop
# ====================================================

def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, 
                              shuffle=True, num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, 
                              shuffle=False, num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    model = EfficientNetClassifier(pretrained=True)
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    if CFG.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr
        )
    elif CFG.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, eps=CFG.eps
        )


    criterion = nn.CrossEntropyLoss()

    best_score = 0.
    best_loss = np.inf



    for epoch in range(CFG.epochs):
        start_time = time.time()
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds[CFG.target_col].values

        if CFG.scheduler == 'plateau':
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        score = get_score(valid_labels, preds.argmax(1))
        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Accuracy: {score}')
        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 'preds': preds}, OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')
    
    check_point = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')
    valid_folds[[str(c) for c in range(5)]] = check_point['preds']
    valid_folds['preds'] = check_point['preds'].argmax(1)

    return valid_folds


# ====================================================
# main function
# ====================================================

def main():

    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.5f}')
    
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(folds, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)

# Load training data
train = pd.read_csv('cassava-leaf-disease-classification/train.csv')
# Split into folds for cross validation - we used the same split for all the models we trained!
folds = train.merge(
    pd.read_csv("validation_data.csv")[["image_id", "fold"]], on="image_id")


if __name__ == '__main__':
    main()
