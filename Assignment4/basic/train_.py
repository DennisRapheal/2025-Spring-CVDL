import os, glob, random
from argparse import Namespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger  # ← 改用 CSVLogger

from utils.dataset_utils import PromptDataset, _numeric_sort
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.metrics import calc_psnr
from net.model import PromptIR
from options import options as opt

# ---------------------- LightningModule ---------------------- #
class PromptIRModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 2e-4,
        warmup_epochs: int = 15,
        max_epochs: int = 150,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    # ---- train ----
    def training_step(self, batch, batch_idx):
        (_, _), de, cl = batch
        out = self(de)
        loss = self.loss_fn(out, cl)

        # 記錄到 CSV
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ---- val ----
    def validation_step(self, batch, batch_idx):
        (_, _), de, cl = batch
        out = self(de)

        loss = self.loss_fn(out, cl)
        psnr = calc_psnr(out, cl)

        # 記錄到 CSV
        self.log_dict(
            {"val/loss": loss, "val/psnr": psnr},
            prog_bar=True,
            sync_dist=True,
        )

    # ---- opt & lr sched ----
    def configure_optimizers(self):
        opt = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        sched = LinearWarmupCosineAnnealingLR(
            optimizer=opt,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.trainer.max_epochs
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1,
            },
        }

# ---------------------- 切資料 ---------------------- #
def build_dataloaders(opt: Namespace):
    """把 train_dir/degraded/* 依 0.9 / 0.1 切出 train / val 路徑，再包 Loader"""
    de_dir = os.path.join(opt.train_dir, "degraded")
    all_de = _numeric_sort(glob.glob(os.path.join(de_dir, "*.png")))
    if not all_de:
        raise RuntimeError(f"No images in {de_dir}")

    idx = list(range(len(all_de)))
    random.seed(42)
    random.shuffle(idx)

    v_len = int(len(idx) * 0.1)
    val_idx = set(idx[:v_len])
    train_paths = [all_de[i] for i in idx]
    val_paths = [all_de[i] for i in val_idx]

    train_ds = PromptDataset(train_paths,
                             patch_size=opt.patch_size,
                             is_train=True)
    val_ds = PromptDataset(val_paths,
                           patch_size=opt.patch_size,
                           is_train=False)

    train_ld = DataLoader(train_ds, batch_size=opt.batch_size,
                          shuffle=True, num_workers=opt.num_workers,
                          pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=8,
                        shuffle=False, num_workers=4, pin_memory=True)
    return train_ld, val_ld

# -------------------------- main -------------------------- #
def main():
    # ----- logger -----
    logger = CSVLogger(save_dir="logs", name="promptir")  # ← 改用 CSVLogger，儲存到 logs/promptir

    # ----- dataloader -----
    train_ld, val_ld = build_dataloaders(opt)

    # ----- model & ckpt -----
    model = PromptIRModel(lr=opt.lr)
    ckpt_cb = ModelCheckpoint(dirpath=opt.ckpt_dir,
                              mode="max", save_top_k=1, every_n_epochs=1)

    # ----- trainer -----
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
        logger=logger,
        callbacks=[ckpt_cb],
        precision="16-mixed"
    )
    trainer.fit(model, train_ld, val_ld)

if __name__ == "__main__":
    main()