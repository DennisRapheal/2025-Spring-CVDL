import os
import argparse
import json
import zipfile

import torch
import numpy as np
from tqdm import tqdm

from model import get_model
from pycocotools import mask as mask_utils
from utils import get_test_loader, encode_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run instance segmentation inference and generate COCO-style outputs."
    )
    parser.add_argument('--img_dir', type=str, default='./data/test_release',
                        help='Path to the test image directory')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        help='Model backbone type (should match the trained model)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model .pth file')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--with_train_map', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./result',
                        help='Directory to save pred.json and pred.zip')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Score threshold for filtering predictions')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                        help='Threshold to binarize mask outputs')
    parser.add_argument('--mask_size', type=int, default=512,
                        help='Resolution (height and width) for resizing images and masks')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes (including background) the model was trained on')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computation device to use (e.g., "cuda" or "cpu")')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load the model architecture and weights
    model = get_model(
        num_classes=args.num_classes,
        model_type=args.model_type,
        with_train_map=args.with_train_map
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    # Load test dataset
    test_loader = get_test_loader(batch_size=args.batch_size, root=args.img_dir)
    print(f"Test images: {len(test_loader.dataset)}")

    output_list = []

    # Inference loop
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Inference"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, img_id in zip(outputs, image_ids):
                # GPU filter by threshold
                keep = output['scores'] >= args.threshold
                scores = output['scores'][keep].cpu()
                labels = output['labels'][keep].cpu()
                masks = output['masks'][keep, 0].cpu()   # (N, H, W)
                boxes = output['boxes'][keep].cpu()      # (N, 4)

                for score, mask, label, box in zip(scores, masks, labels, boxes):
                    # Convert box to xywh
                    x1, y1, x2, y2 = box.tolist()
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    # Binarize and encode mask
                    mask_bin = (mask > args.mask_threshold).numpy().astype(np.uint8)
                    seg_rle = encode_mask(mask_bin)

                    output_list.append({
                        "image_id": int(img_id),
                        "bbox": [float(v) for v in bbox],
                        "score": float(score),
                        "category_id": int(label),
                        "segmentation": seg_rle
                    })

    # Save pred.json
    json_path = os.path.join(args.output_dir, 'pred.json')
    with open(json_path, 'w') as f:
        json.dump(output_list, f)
    print(f"Saved prediction results to {json_path}")

    # Save pred.zip
    zip_path = os.path.join(args.output_dir, 'pred.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname='test-results.json')
    print(f"Saved zipped prediction results to {zip_path}")


if __name__ == '__main__':
    main()
