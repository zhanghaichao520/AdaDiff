export CUDA_VISIBLE_DEVICES=6
python main.py --model GPT2 --dataset CDs_and_Vinyl --quant_method rvq

export CUDA_VISIBLE_DEVICES=5
python main.py --model RPG --dataset Sports_and_Outdoors --quant_method pq

export CUDA_VISIBLE_DEVICES=7
python main.py --model TIGER --dataset Baby --quant_method rqvae