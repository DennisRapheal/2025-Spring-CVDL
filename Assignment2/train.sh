# python train.py --model_name resnext50_32x4d --model_type resnext50_32x4d --batch_size 32 --num_epochs 10
# python train.py --model_name resnext101_32x8d --model_type resnext101_32x8d --batch_size 32 --num_epochs 30

# python inference.py --model_type resnext50_32x4d --model_path ./checkpoints/resnext50_32x4d.pth --output_dir resnext50_32x4d --batch_size 32
# python inference.py --model_type resnext50_32x4d --model_path ./checkpoints/resnext50_32x4d.pth --output_dir resnext50_32x4d_threshold_07 --batch_size 32 --threshold 0.7

python train.py --model_name resnext50_32x4d_hard_transform --model_type resnext50_32x4d --from_pretrained ./checkpoints/resnext50_32x4d.pth --batch_size 32 --num_epochs 30
python train.py --model_name resnext50_32x4d_hard_transform_resize256 --model_type resnext50_32x4d --from_pretrained ./checkpoints/resnext50_32x4d.pth --batch_size 16 --num_epochs 25

python train.py --model_name resnext101_32x8d_hard_transform_resize256 --model_type resnext101_32x8d --batch_size 16 --num_epochs 35
python train.py --model_name fasterrcnn_resnet50_fpn_v2 --model_type fasterrcnn_resnet50_fpn_v2 --batch_size 16 --num_epochs 15
python train.py --model_name fasterrcnn_v2_hard_transform --model_type fasterrcnn_resnet50_fpn_v2 --transform_type hard --from_pretrained ./checkpoints/fasterrcnn_resnet50_fpn_v2.pth --batch_size 32 --num_epochs 15

python inference.py --model_type resnext50_32x4d --model_path ./checkpoints/resnext50_32x4d_hard_transform_resize256.pth --output_dir resnext50_32x4d_hard_transform_resize256 --batch_size 32 --threshold 0.7
python inference.py --model_type fasterrcnn_resnet50_fpn_v2 --model_path ./checkpoints/fasterrcnn_resnet50_fpn_v2.pth --output_dir fasterrcnn_resnet50_fpn_v2 --batch_size 32 --threshold 0.7