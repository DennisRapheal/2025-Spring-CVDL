# python inference.py \
#   --img_dir ./sample-image \
#   --model_type resnet50_v2 \
#   --model_path ./checkpoints/resnet50_v2_trainmap_maskrcnn.pth \
#   --batch_size 1 \
#   --output_dir ./sample-image_result \
#   --threshold 0.7 \
#   --mask_threshold 0.3


CUDA_VISIBLE_DEVICES=1 python inference.py \
  --img_dir ./data/test_release \
  --model_type resnet50_v2 \
  --model_path ./checkpoints/v2_b2_pure_aux_maskrcnn.pth \
  --batch_size 2 \
  --output_dir ./result \
  --threshold 0.1 \
  --mask_threshold 0.6 \
  --with_train_map

python inference.py \
  --img_dir ./sample-image \
  --model_type resnet50 \
  --model_path ./checkpoints/v1_b4_maskrcnn.pth \
  --batch_size 1 \
  --output_dir ./sample-image_result \
  --threshold 0.0 \
  --mask_threshold 0.5

python coco_eval.py
python vis_sample.py
