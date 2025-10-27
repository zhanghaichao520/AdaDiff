export CUDA_VISIBLE_DEVICES=3
python main.py --model TIGER --dataset Sports_and_Outdoors --quant_method rvq

export CUDA_VISIBLE_DEVICES=1
python main.py --model RPG --dataset Toys_and_Games --quant_method pq

export CUDA_VISIBLE_DEVICES=2
python main.py --model TIGER --dataset Toys_and_Games --quant_method rqvae