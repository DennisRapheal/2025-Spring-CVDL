import torch
import math

def calc_psnr(restored, target):
    """
    restored / target : [B,3,H,W], 0-1 float
    回傳平均 PSNR（scalar - float）
    """
    mse = torch.clamp((restored - target) ** 2, 1e-10, 1.).mean(dim=[1,2,3])
    psnr = 10 * torch.log10(1. / mse)
    return psnr.mean().item()
