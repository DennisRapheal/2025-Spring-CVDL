import csv

csv_path = './weights/resnext50_32x4d/training_log.csv'

# 欲列印的欄位
target_columns = [
    "Epoch",
    "Train Loss",
    "Train mAP",
    "Train mAP@0.5",
    "Val mAP",
    "Val mAP@0.5",
    "LR"
]

try:
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

        # 分別列印每個欄位的所有值
        for col in target_columns:
            print(f"{col}:")
            for row in data:
                print(row[col])
            print("-" * 30)  # 分隔線讓輸出更清楚

except FileNotFoundError:
    print(f"找不到檔案：{csv_path}")
except Exception as e:
    print(f"發生錯誤：{e}")
