
export CUDA_VISIBLE_DEVICES=1
python amazon_text_emb.py --dataset Baby \
    --model_name_or_path /data/jwna230/peiyu/LLM-Models/Qwen/Qwen3-Embedding-8B \
    --plm_name qwen