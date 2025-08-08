
export CUDA_VISIBLE_DEVICES=2
python amazon_text_emb.py --dataset Baby \
    --model_name_or_path /mnt/disk9T/zj/projects/peiyu/LLM_Models/Qwen/Qwen3-Embedding-8B \
    --plm_name qwen