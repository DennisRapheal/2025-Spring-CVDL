import os
import argparse
import torch
import json
import pandas as pd
import zipfile
from tqdm import tqdm
from model import get_model
from utils import get_test_loader, set_batch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./data/test', help='Path to test image directory')
    parser.add_argument('--model_type', type=str, default='resnext50_32x4d', help='Path to test image directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='./result', help='Directory to save pred.json and pred.csv')
    parser.add_argument('--threshold', type=float, default=0.3, help='Score threshold for filtering predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def run_inference(args):
    device = torch.device(args.device)
    set_batch(args.batch_size)
    # Load model
    model = get_model(model_type=args.model_type, weights_pth=args.model_path)
    model.to(device)
    model.eval()

    # Load test data
    test_loader = get_test_loader()

    task1_output = []  # For pred.json
    task2_output = []  # For pred.csv

    for images, image_ids, original_sizes in tqdm(test_loader):
        images = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images)

        for img_id, output, (orig_w, orig_h) in zip(image_ids, outputs, original_sizes):
            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()

            # 轉換比例（resize 是 256x256）
            scale_x = orig_w / 256
            scale_y = orig_h / 256

            # Task 1
            for box, score, label in zip(boxes, scores, labels):
                if score < args.threshold:
                    continue
                x1, y1, x2, y2 = box.tolist()
                # 還原 bbox
                x1 *= scale_x
                x2 *= scale_x
                y1 *= scale_y
                y2 *= scale_y
                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    continue

                task1_output.append({
                    "image_id": int(img_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                    "category_id": int(label)
                })

            # Task 2 保持不變，但要用還原後的 box 排序
            keep = scores > args.threshold
            boxes = boxes[keep]
            labels = labels[keep]
            if len(boxes) == 0:
                pred_label = "-1"
            else:
                x_coords = boxes[:, 0] * scale_x
                sorted_indices = x_coords.argsort()
                sorted_labels = labels[sorted_indices]
                pred_label = ''.join(str(l.item() - 1) for l in sorted_labels)

            task2_output.append({
                "image_id": int(img_id),
                "pred_label": pred_label
            })

    # Save results
    os.makedirs(f'{args.output_dir}', exist_ok=True)

    json_path = os.path.join(args.output_dir, "pred.json")
    with open(json_path, "w") as f:
        json.dump(task1_output, f)
    print(f"Task1 (Detection) saved to {json_path}")

    csv_path = os.path.join(args.output_dir, "pred.csv")
    pd.DataFrame(task2_output).to_csv(csv_path, index=False)
    print(f"Task2 (Recognition) saved to {csv_path}")
    
    zip_path = os.path.join(args.output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(json_path, arcname="pred.json")
        zipf.write(csv_path, arcname="pred.csv")
    print(f"Zipped submission saved to {zip_path}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
