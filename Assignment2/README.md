# NYCU Computer Vision 2025 Spring HW2

StudentID: 111550006

Name: 林庭寪

## Introduction

This project tackles digit recognition in natural scene images using the Faster R-CNN framework. The task is divided into two parts:

* **Task 1** detects and classifies each individual digit in the image (object detection).
* **Task 2** recognizes the full digit sequence as a whole (sequence recognition).

The model is implemented using PyTorch's Faster R-CNN and trained on a COCO-style digit dataset. To boost performance, strong data augmentation techniques were applied. Evaluation is based on mean Average Precision (mAP), and a confidence score threshold is used during inference to refine predictions.

## How to Install

```bash
pip install requirements.txt
```

* see `train.sh` for training and inferencing

## Performance Snapshot

![1744828689125](image/README/1744828689125.png)
