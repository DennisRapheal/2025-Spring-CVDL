cd ..
# choose a model you want to visualize
python inference.py \
  --img_dir ./sample-image \
  --model_type resnet50 \
  --model_path ./checkpoints/v1_b4_maskrcnn.pth \
  --batch_size 1 \
  --output_dir ./sample-image_result \
  --threshold 0.0 \
  --mask_threshold 0.5

cd verify

# get ground truth for coco_eval.py
python get_gt_json.py

# make sure the prediction is valid
python coco_eval.py

# visualize segmentation
python vis_sample.py