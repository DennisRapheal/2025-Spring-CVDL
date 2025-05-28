#!/usr/bin/env python3
# compare_training_curves.py
# ---------------------------------------------------------------
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_A = "./../TrainingLog/v2_b2_elastic.csv"
CSV_B = "./../TrainingLog/v2_b2_pure_aux.csv"
OUT_DIR = Path("./plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_a = pd.read_csv(CSV_A)
df_b = pd.read_csv(CSV_B)

if "Epoch" not in df_a.columns or "Epoch" not in df_b.columns:
    raise ValueError("兩個 CSV 都必須包含 'Epoch' 欄位")

metrics = [
    ("Train Loss",      "Training Loss ↓"),
    ("Train mAP",       "Train mAP ↑"),
    ("Train mAP@0.5",   "Train mAP@0.5 ↑"),
    ("Val mAP",         "Val mAP ↑"),
    ("Val mAP@0.5",     "Val mAP@0.5 ↑"),
]

for col, nice_name in metrics:
    if col not in df_a.columns or col not in df_b.columns:
        print(f"⚠️  找不到欄位 {col}，跳過。")
        continue

    plt.figure(figsize=(6,4))
    plt.plot(df_a["Epoch"], df_a[col], label="w/o train_map")
    plt.plot(df_b["Epoch"], df_b[col], label="w/ train_map")

    plt.xlabel("Epoch")
    plt.ylabel(nice_name)
    plt.title(f"{nice_name} – w/o Train Map vs  w/ Train Map")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = OUT_DIR / f"{col.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

print(f"\n所有圖已輸出到：{OUT_DIR.resolve()}")
