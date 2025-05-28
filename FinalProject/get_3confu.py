import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 讀檔
df = pd.read_csv("/home/ccwang/dennis/dennislin0906/cvdl-final/outputs/effv2-m/oof_val_df.csv")


# 檔案讀完後立刻處理 label 和 preds
df["label"] = df["label"].replace(4, 3)
df["preds"] = df["preds"].replace(4, 3)

# 類別名稱對應表（注意：Healthy 是 3）
label_name_map = {
    0: "CBB",
    1: "CBSD",
    2: "CGM",
    3: "Healthy"
}

labels_used = sorted(df["label"].unique())   # 應該會是 [0, 1, 2, 3]
target_names = [label_name_map[i] for i in labels_used]


# 輸出資料夾
save_dir = "cm_jpg"
os.makedirs(save_dir, exist_ok=True)

# 儲存文字報表
txt_path = os.path.join(save_dir, "metrics.txt")
with open(txt_path, "w", encoding="utf-8") as fout:

    for fold, g in df.groupby("fold"):
        y_true = g["label"]
        y_pred = g["preds"]
        cm     = confusion_matrix(y_true, y_pred, labels=labels_used)
        acc    = accuracy_score(y_true, y_pred)

        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                    xticklabels=target_names,
                    yticklabels=target_names)
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
            target_names=target_names
        ))
        fout.write("\n")

    # 總體結果
    y_true_all = df["label"]
    y_pred_all = df["preds"]
    cm_total   = confusion_matrix(y_true_all, y_pred_all, labels=labels_used)
    acc_total  = accuracy_score(y_true_all, y_pred_all)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_total, annot=True, fmt="d", cbar=False,
                xticklabels=target_names,
                yticklabels=target_names)
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
        target_names=target_names
    ))
    fout.write("\n")

print(f"✅ 所有文字報表已寫入：{txt_path}")
