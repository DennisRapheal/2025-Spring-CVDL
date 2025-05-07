import os
import numpy as np
import tifffile
from scipy.ndimage import center_of_mass, gaussian_filter, label, binary_dilation, binary_erosion

'''
 get heat map and boundary map
'''
def load_and_merge_masks(folder_path):
    mask = None
    instance_id = 1
    for class_idx in range(1, 5):
        class_mask_path = os.path.join(folder_path, f'class{class_idx}.tif')
        if os.path.exists(class_mask_path):
            class_mask = tifffile.imread(class_mask_path)
            binary_mask = (class_mask > 0).astype(np.uint8)
            labeled, num_features = label(binary_mask)
            labeled[labeled > 0] += (instance_id - 1)
            if mask is None:
                mask = labeled
            else:
                mask = np.maximum(mask, labeled)
            instance_id = mask.max() + 1
    return mask

def generate_center_heatmap(mask, sigma=5):
    heatmap = np.zeros_like(mask, dtype=np.float32)
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]
    
    for id in instance_ids:
        binary_mask = (mask == id).astype(np.uint8)
        cy, cx = center_of_mass(binary_mask)
        if not np.isnan(cy) and not np.isnan(cx):
            cy = int(round(cy))
            cx = int(round(cx))
            if 0 <= cy < heatmap.shape[0] and 0 <= cx < heatmap.shape[1]:
                heatmap[cy, cx] = 1.0

    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def generate_boundary_map(mask):
    boundary = np.zeros_like(mask, dtype=np.uint8)
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]

    for id in instance_ids:
        binary = (mask == id).astype(np.uint8)
        dilated = binary_dilation(binary)
        eroded = binary_erosion(binary)
        edge = dilated ^ eroded
        boundary = np.maximum(boundary, edge)
    
    return boundary

def process_dataset(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_ids = os.listdir(root_dir)
    
    for image_id in image_ids:
        folder_path = os.path.join(root_dir, image_id)
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Processing {image_id}...")
        
        mask = load_and_merge_masks(folder_path)
        if mask is None:
            print(f"Warning: No valid masks found for {image_id}. Skipping...")
            continue
        
        heatmap = generate_center_heatmap(mask, sigma=5)
        boundary = generate_boundary_map(mask)

        np.save(os.path.join(output_dir, f"{image_id}_center_heat_map.npy"), heatmap)
        np.save(os.path.join(output_dir, f"{image_id}_boundary_map.npy"), boundary)

process_dataset(
    root_dir='data/train',
    output_dir='data/train_maps'
)