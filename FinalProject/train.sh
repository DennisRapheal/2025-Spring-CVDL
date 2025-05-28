clear
# model_1 384*384 efficientnet_b4
# CUDA_VISIBLE_DEVICES=5,6,7 python train.py \
#     --weight_decay 1e-3 \
#     --amp \
#     --batch_size 12 \
#     --epochs 20

# model_3, tf_efficientnetv2_xl
# CUDA_VISIBLE_DEVICES=5,6 python train.py \
#     --eff_only \
#     --name model_3 \
#     --img_size 512 \
#     --weight_decay 1e-3 \
#     --amp \
#     --batch_size 32 \
#     --epochs 30

# model_2 384*384, tf_efficientnetv2_l
# CUDA_VISIBLE_DEVICES=0,7 python train.py \
#     --name model_2 \
#     --weight_decay 1e-2 \
#     --amp \
#     --batch_size 4 \
#     --epochs 25 \
#     --lr 1e-4

# model_4, tf_efficientnetv2_xl full train data
# CUDA_VISIBLE_DEVICES=2,3 python train.py \
#     --val_ratio 0.1 \
#     --eff_only \
#     --name model_4 \
#     --img_size 512 \
#     --weight_decay 1e-3 \
#     --lr 1e-4 \
#     --amp \
#     --batch_size 16 \
#     --epochs 20

# model_5 384*384, tf_efficientnetv2_l full train data
# CUDA_VISIBLE_DEVICES=0,3 python train.py \
#     --name model_5 \
#     --weight_decay 1e-4 \
#     --amp \
#     --batch_size 8 \
#     --epochs 25 \
#     --img_size 384 \
#     --lr 1e-4

# model_6 384*384, tf_efficientnetv2_l full train data
# CUDA_VISIBLE_DEVICES=3,5,7 python train.py \
#     --name model_6 \
#     --weight_decay 1e-4 \
#     --amp \
#     --batch_size 8 \
#     --epochs 150 \
#     --img_size 384 \
#     --lr 1e-4

# model_7 384*384, tf_efficientnetv2_l full train data
# CUDA_VISIBLE_DEVICES=0,3,5,7 python train.py \
#     --name model_7 \
#     --weight_decay 1e-2 \
#     --amp \
#     --batch_size 16 \
#     --epochs 150 \
#     --img_size 384 \
#     --lr 1e-5

# with mix up, all mix up lr = 1e-4, AdamW, weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=0,3,5,6 python train5fold.py


CUDA_VISIBLE_DEVICES=0,3,5,6 python train4label.py