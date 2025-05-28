# NYCU Computer Vision 2025 Spring HW3

StudentID: 111550006

Name: 林庭寪

## Introduction

### Task Overview

This assignment focuses on the challenging task of blind image restoration, where the goal is to recover clean images from degraded inputs affected by rain or snow. Unlike traditional restoration tasks that handle only a single type of corruption, this task requires designing a single model capable of restoring multiple degradation types without prior knowledge of the specific corruption present in each input. The dataset includes paired degraded and clean images for both rain and snow conditions, and model performance is evaluated using Peak Signal-to-Noise Ratio (PSNR). As external data and pretrained weights are not allowed, we adopt and improve the PromptIR architecture, a recent model that introduces tunable prompts to encode degradation-specific features.

### Method Overview

I use PromptIR as the backbone model. PromptIR is a novel transformer-based framework for blind image restoration, where the model doesn't rely on explicit knowledge of degradation types. Instead, it learns to adaptively guide restoration via learned prompts. These prompts act like conditioning signals generated on-the-fly to steer the model in different stages of the restoration process. It builds on hierarchical transformers and introduces Prompt Blocks (Prompt Generation Module PGM and Prompt Interaction Module PIM) that adaptively generate and inject restoration cues at different stages of decoding.

To improve the adaptability and expressiveness of prompt features in PromptIR, I introduced two key modifications: adjusting the prompt length and enhancing the Prompt Generation Block (PromptGenBlock) with multi-scale feature processing.

## How to Install

Refered to [PromptIR repo](https://github.com/va1shn9v/PromptIR?tab=readme-ov-file).  Or simply run

```bash
pip install requirements.txt
```

## How to Train the Model

See `train.sh` for some script templates for training.


### Run Example

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
  --train_dir ./hw4_realse_dataset/train
  --ckpt_dir ./train_ckpt
  --output_path ./output
  --num_gpus 4
  --epochs 120
  --batch_size 8
  --patch_size 128
```


### Arguments

| Argument          | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| `--cuda`        | GPU device index (default:`0`)                            |
| `--epochs`      | Maximum number of training epochs (default:`120`)         |
| `--batch_size`  | Batch size per GPU (default:`8`)                          |
| `--lr`          | Learning rate for training (default:`2e-4`)               |
| `--patch_size`  | Input patch size for cropped images (default:`128`)       |
| `--de_type`     | Degradation types used (can specify multiple)               |
| `--train_dir`   | Path to the training dataset (should contain `degraded/`) |
| `--output_path` | Directory to save logs and predictions                      |
| `--ckpt_dir`    | Directory to save model checkpoints                         |
| `--num_gpus`    | Number of GPUs for distributed training                     |
| `--num_workers` | Number of dataloader workers (default:`16`)               |
| `--wblogger`    | WandB project name (if logging is enabled)                  |

## How to Inference with Test Data

See `infer.sh` for some script templates for inferencing.

examples:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --ckpt ckpt/epoch=149-step=9999.ckpt \
  --input_dir ./data/test/degraded \
  --output_dir ./runs/predict
```

### Arguments

| Argument          | Description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| `--ckpt`        | Path to the `.ckpt`file saved during training (from `ModelCheckpoint`) |
| `--input_dir`   | Folder containing degraded images (PNG, JPG, BMP, etc.)                    |
| `--output_dir`  | Folder to save restored output (default:`output_pred`)                   |
| `--batch_size`  | Batch size for inference (default:`4`)                                   |
| `--num_workers` | Number of dataloader workers (default:`4`)                               |
| `--half`        | Use half precision (float16) if GPU is available                           |
| `--cuda`        | Set GPU device index (use `--cuda -1`to run on CPU)                      |

### Outputs

After execution, the following will be generated in `--output_dir`:

* `pred_images/`: Folder of restored images
* `pred.npz`: Compressed `.npz` archive of restored image arrays (CHW format)
* `pred.zip`: Zipped version of the `.npz` for easy submission or download

## Visualization

Inference.py will store reconstructed image in ./outputs.

## Performance Snapshot

![](https://file+.vscode-resource.vscode-cdn.net/Users/lintingwei/Desktop/cvdl/Assignment4/lb.png?version%3D1748443550116)
