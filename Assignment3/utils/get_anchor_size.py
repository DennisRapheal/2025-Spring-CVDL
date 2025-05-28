import os
import numpy as np
from pathlib import Path
import skimage.io as sio

all_areas = []

root_dir = Path("./../data/train")
for folder in root_dir.iterdir():
    for cls in range(1, 5):
        path = folder / f"class{cls}.tif"
        if not path.exists():
            continue
        mask = sio.imread(path)
        inst_ids = np.unique(mask)
        for inst_id in inst_ids:
            if inst_id == 0:
                continue
            binary = (mask == inst_id)
            y, x = np.where(binary)
            h = y.max() - y.min() + 1
            w = x.max() - x.min() + 1
            area = h * w
            all_areas.append(area)

side_lengths = np.sqrt(all_areas)
percentiles = np.percentile(side_lengths, [10, 30, 50, 70, 90])
anchor_sizes = tuple((int(s),) for s in percentiles)

print("recomended anchor sizes:", anchor_sizes)
