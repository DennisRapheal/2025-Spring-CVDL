# NYCU Computer Vision 2025 Spring HW3

StudentID: 111550006

Name: 林庭寪

## Introduction

This project tackles instance segmentation on colored medical images, aiming to identify and segment four distinct cell types. The dataset includes 209 training/validation images and 101 test images in `.tif` format.

The core model is based on  **Mask R-CNN** , using a **ResNet-50 (or ResNet-50 V2)** backbone and a **Feature Pyramid Network (FPN)** for robust multi-scale feature extraction. To improve segmentation of  **cluttered or overlapping cells** , the model optionally includes **auxiliary heads** for predicting **cell center** and  **boundary maps** , inspired by approaches like  **CellPose** . These features are designed to guide the model toward more accurate and well-separated instance masks.

## How to Install

```bash
pip install requirements.txt
```

## How to Train the Model

See `train.sh` for some script templates for training.

## How to Inference with Test Data

See `infer.sh` for some script templates for inferencing.

## Veirfy Format and Visualization

See ./verify for some visual results and 

## Performance Snapshot
