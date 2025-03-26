# export PATH=/usr/local/cuda-12.8/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
# export CUDA_HOME=/usr/local/cuda-12.8

# rm -rf ~/.cache/torch/hub

# Trained on cuda-12.8, python 3.12
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# pip install -r requirements.txt

nvcc --version
python -c "import torch; import torchvision; print(f'Torch: {torch.__version__}, TorchVision: {torchvision.__version__}')"

python train.py --model_name  seresnext101_32x8d --model_type  seresnext101_32x8d --num_epochs 100 --schedulerT_0 100 --schedulerT_mult 2 --batch_size 128 --mix_prob 0.4 && python inference.py --model_name  seresnext101_32x8d --model_type  seresnext101_32x8d --batch_size 32