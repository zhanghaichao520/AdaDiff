export CUDA_VISIBLE_DEVICES=0
python main.py --model TIGER --dataset Toys_and_Games --quant_method rkmeans

export CUDA_VISIBLE_DEVICES=1
python main.py --model GPT2 --dataset Toys_and_Games --quant_method rqvae

python main.py --model GPT2 --dataset Musical_Instruments --quant_method rqvae