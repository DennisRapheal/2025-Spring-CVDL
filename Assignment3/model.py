import torch
from torch import nn
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_Weights,          # v1 權重列舉
    MaskRCNN_ResNet50_FPN_V2_Weights        # v2 權重列舉
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


class ExtraHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, name: str):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
        self.name = name  # for debug or loss logging

    def forward(self, x):
        return self.head(x)


def get_model(
    num_classes: int,
    model_type: str = "resnet50",
    with_train_map: bool = False,
    customed_anchor: bool = False
):
    """
    建立 Mask R‑CNN 模型並替換 heads 以符合自訂類別數。

    Args:
        num_classes (int): 包含背景的總類別數 (背景 + N 物件)。
        model_type  (str): 'resnet50' 或 'resnet50_v2'。
        with_train_map (bool): 是否包含 center / boundary head。
        customed_anchor (bool): 是否使用自訂 anchor 設定。

    Returns:
        torchvision.models.detection.MaskRCNN
    """
    # 1. 載入指定 backbone 與預訓練權重
    if model_type == "resnet50":
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights, progress=True)
    elif model_type == "resnet50_v2":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=True)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}', expected 'resnet50' or 'resnet50_v2'"
        )

    # 2. 取出原本 ROI heads 的輸入通道
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # 3. 替換 Box predictor (分類 + bbox regression)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # 4. 替換 Mask predictor
    hidden_dim = 256  # MaskRCNN 預設 hidden dim
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels_mask, hidden_dim, num_classes
    )

    if with_train_map:
        # 假設我們使用的是 mask feature 的輸出通道
        # 可替換成其他 feature，例如 model.backbone.out_channels
        model.center_head = ExtraHead(in_channels_mask, 1, name="center")       # binary output
        model.boundary_head = ExtraHead(in_channels_mask, 1, name="boundary")   # binary output

    if customed_anchor:
        anchor_sizes = ((16,), (19,), (23,), (29,), (37,))  # 可外部指定
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )

    return model
