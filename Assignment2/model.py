from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch

num_classes = 11

def get_model(model_type='resnext50_32x4d', weights_pth=None):
    model = None

    if model_type in ['resnet18', 'resnet50', 'resnet101', 'resnext50_32x4d', 'resnext101_32x8d']:
        backbone = resnet_fpn_backbone(model_type, pretrained=True)
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif model_type == 'fasterrcnn_resnet50_fpn_v2':
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported torchvision model_type: {model_type}")

    if weights_pth is not None:
        model.load_state_dict(torch.load(weights_pth))

    return model