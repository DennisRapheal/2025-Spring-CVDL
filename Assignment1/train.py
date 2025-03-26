from model import get_model
from utils import get_train_loader, get_val_loader, cutmix_criterion, cutmix_data, mixup_criterion, mixup_data, set_batch
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import os
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")

    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="Name of the model architecture (e.g., resnet50, resnet101, regnet_y_8gf, etc.)")

    parser.add_argument("--model_type", type=str, default="resnet50",
                        help="Type of the base model (e.g., resnet50, regnet_y_8gf)")

    parser.add_argument("--mix_prob", type=float, default=0.4,
                        help="Probability of mixing augmentations")

    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")

    parser.add_argument("--schedulerT_0", type=int, default=100,
                        help="Initial T_0 value for the scheduler")

    parser.add_argument("--schedulerT_mult", type=int, default=2,
                        help="Multiplier for the scheduler T")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    # params
    model_name = args.model_name
    model_type = args.model_type
    mix_prob = args.mix_prob
    num_epochs = args.num_epochs
    schedulerT_0 = args.schedulerT_0
    schedulerT_mult = args.schedulerT_mult
    batch_size = args.batch_size

    print(f"Model Name: {model_name}")
    print(f"Model Type: {model_type}")
    print(f"Mix Probability: {mix_prob}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Scheduler T_0: {schedulerT_0}")
    print(f"Scheduler T_mult: {schedulerT_mult}")
    print(f"Batch Size: {batch_size}")

    # 確保 weights 目錄存在
    os.makedirs("./weights", exist_ok=True)

    # 初始化模型與裝置
    model = get_model(model_type=model_type)
    print("model is loaded!!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 移動模型到 GPU/CPU

    train_loader = get_train_loader(batch_size)
    print("train data loaded!!")
    val_loader = get_val_loader(batch_size)
    print("val data loaded!!")

    # 設定損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-2)

    # 設定 CosineAnnealingLR 調度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=schedulerT_0, T_mult=schedulerT_mult
    )
    history = []  # 存儲結果
    
    # model.unfreeze_head()

    max_val_acc = 0.0
    print("-------------training start-------------")

    for ep in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            prob = np.random.rand()
            outputs = None
            loss = None

            if prob < mix_prob:
                alpha = np.random.uniform(0.4, 0.6)
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif prob < mix_prob * 2:
                alpha = np.random.uniform(0.8, 1.2)
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha)
                outputs = model(inputs)
                loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # 計算 validation accuracy
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        current_lr = optimizer.param_groups[0]['lr']
        val_acc = 100. * val_correct / val_total

        print(f"Epoch [{ep+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {current_lr: .10f}")
        history.append([ep + 1, train_loss, train_acc, val_acc, current_lr])

        if val_acc > max_val_acc:
            torch.save(model.state_dict(), f"./weights/{model_name}.pth")

        # if ep == 10:
        #     model.unfreeze_layer4()

        # if ep == 20:
        #     model.unfreeze_layer3()  # Consider adjusting the timing

        # if ep == 30:
        #     model.unfreeze_base_model()

        scheduler.step()  # **Fixed placement of scheduler step**

    # 儲存訓練歷史數據
    df = pd.DataFrame(history, columns=['Epoch', 'Train Loss', 'Train Acc', 'Val Acc', 'LR'])
    df.to_csv(f'./weights/{model_name}_training_log.csv', index=False)
    
