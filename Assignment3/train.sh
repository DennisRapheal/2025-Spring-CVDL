# python train.py --model_name resnet50_v2_baseline --model_type resnet50_v2 --epochs 100 --batch_size 4
# python train.py --model_name resnet50_baseline --model_type resnet50 --epochs 100 --batch_size 4


#alpha 30 sigma 12 # no transform
python train.py --model_name v1_b1 --model_type resnet50 --epochs 80 --batch_size 1

python train.py --model_name v2_b2_aux --with_train_map --model_type resnet50_v2 --epochs 80 --batch_size 2
#alpha 30 sigma 12 only elastic
CUDA_VISIBLE_DEVICES=1 
python train.py --model_name v1_b1_elastic --model_type resnet50 --epochs 50 --batch_size 1
python train.py --model_name v1_b4_elastic --model_type resnet50 --epochs 100 --batch_size 4

#alpha 30 sigma 12 # elastic
python train.py --model_name v1_b2_elastic --model_type resnet50 --epochs 100 --batch_size 2

#alpha 30 sigma 12 # elastic
python train.py --model_name v2_b2_elastic --model_type resnet50_v2 --epochs 80 --batch_size 2

#alpha 30 sigma 12, all transform, 超爛
CUDA_VISIBLE_DEVICES=4 
python train.py --model_name v2_b4_elastic --model_type resnet50_v2 --pretrained_pth ./checkpoints/v2_b4_elastic_bone.pth --epochs 100 --batch_size 4 --lr 3e-5
# python train.py --model_name resnet50_baseline --model_type resnet50 --epochs 100 --batch_size 4
# python train.py --model_name resnet50_v2_nocrop --model_type resnet50_v2 --epochs 70 --batch_size 4

CUDA_VISIBLE_DEVICES=1 python train.py --model_name v2_b2_preaux --model_type resnet50_v2 --epochs 30 --batch_size 2 && CUDA_VISIBLE_DEVICES=1 python train.py --model_name v2_b2_aux --with_train_map --model_type resnet50_v2 --epochs 80 --batch_size 2 --pretrained_pth /home/ccwang/dennis/dennislin0906/cvdl-hw3/checkpoints/v2_b2_preaux_maskrcnn.pth

CUDA_VISIBLE_DEVICES=3 python train.py --model_name v2_b2_pure_aux --with_train_map --model_type resnet50_v2 --epochs 80 --batch_size 2

CUDA_VISIBLE_DEVICES=3 python train.py --model_name v2_b2_customed_anchor --customed_anchor --model_type resnet50_v2 --epochs 50 --batch_size 4