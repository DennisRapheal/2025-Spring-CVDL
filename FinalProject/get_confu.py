import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 讀檔
df = pd.read_csv("/home/ccwang/dennis/dennislin0906/cvdl-final/outputs/effvs-s/oof_val_df.csv")
n_classes = 5                      # Cassava 5 類

save_dir = "cm_jpg"
os.makedirs(save_dir, exist_ok=True)

txt_path = os.path.join(save_dir, "metrics.txt")
with open(txt_path, "w", encoding="utf-8") as fout:

    for fold, g in df.groupby("fold"):
        y_true = g["label"]
        y_pred = g["preds"]
        cm     = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
        acc    = accuracy_score(y_true, y_pred)

        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                    xticklabels=np.arange(n_classes),
                    yticklabels=np.arange(n_classes))
        plt.title(f"Fold {fold}; ACC = {acc:.4f}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"fold{fold}_cm.jpg"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        fout.write(f"\n===== Fold {fold} (ACC={acc:.4f}) =====\n")
        fout.write(classification_report(
            y_true,
            y_pred,
            digits=4,
            target_names=["CBB", "CBSD", "CGM", "CMD", "Healthy"]
        ))
        fout.write("\n")

    y_true_all = df["label"]
    y_pred_all = df["preds"]
    cm_total   = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(n_classes))
    acc_total  = accuracy_score(y_true_all, y_pred_all)


    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_total, annot=True, fmt="d", cbar=False,
                xticklabels=np.arange(n_classes),
                yticklabels=np.arange(n_classes))
    plt.title(f"Overall OOF; ACC = {acc_total:.4f}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "overall_cm.jpg"),
                dpi=300, bbox_inches="tight")
    plt.close()

    fout.write(f"\n===== Overall OOF (ACC={acc_total:.4f}) =====\n")
    fout.write(classification_report(
        y_true_all,
        y_pred_all,
        digits=4,
        target_names=["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    ))
    fout.write("\n")

print(f"所有文字報表已寫入：{txt_path}")
