import csv

csv_path = '/home/ccwang/dennis/dennislin0906/cvdl-hw2/weights/resnext50_32x4d_hard_transform_resize256/training_log.csv'

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

        for col in target_columns:
            print(f"{col}:")
            for row in data:
                print(row[col])
            print("-" * 30)

except FileNotFoundError:
    print(f"找不到檔案：{csv_path}")
except Exception as e:
    print(f"發生錯誤：{e}")
