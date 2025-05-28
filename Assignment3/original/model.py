import torch
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_Weights,          # v1 權重列舉
    MaskRCNN_ResNet50_FPN_V2_Weights        # v2 權重列舉
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn   import MaskRCNNPredictor


def get_model(num_classes: int, model_type: str = "resnet50"):
    """
    建立 Mask R‑CNN 模型並替換 heads 以符合自訂類別數。

    Args:
        num_classes (int): 包含背景的總類別數 (背景 + N 物件)。
        model_type  (str): 'resnet50' 或 'resnet50_v2'。

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
    in_features_box  = model.roi_heads.box_predictor.cls_score.in_features
    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # 3. 替換 Box predictor (分類 + bbox regression)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # 4. 替換 Mask predictor
    hidden_dim = 256  # MaskRCNN 預設 hidden dim
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels_mask, hidden_dim, num_classes
    )

    return model
