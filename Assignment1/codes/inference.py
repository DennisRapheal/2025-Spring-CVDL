import torch
import pandas as pd
import os
import zipfile
from model import get_model
from utils import get_test_loader
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")

    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="Name of the model architecture (e.g., resnet50, resnet101, regnet_y_8gf, etc.)")
    parser.add_argument("--model_type", type=str, default="resnet50",
                        help="Type of the base model (e.g., resnet50, regnet_y_8gf)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()
    model_name = args.model_name
    model_type = args.model_type
    batch_size = args.batch_size
    os.makedirs("./weights", exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_type, weights_pth=f'./weights/{model_name}.pth')
    model.to(device)
    model.eval()

    test_loader = get_test_loader(batch_size)

    result = []

    with torch.no_grad():
        for inputs, names in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for img_name, pred in zip(names, predicted.cpu().tolist()):
                img_name = os.path.splitext(img_name)[0]
                result.append([img_name, pred])

    csv_filename = "./weights/prediction.csv"
    zip_filename = f"./weights/{model_name}_result.zip"

    df = pd.DataFrame(result, columns=['image_name', 'pred_label'])
    df.to_csv(csv_filename, index=False)

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_filename, os.path.basename(csv_filename))

    print("Inference 完成!")
