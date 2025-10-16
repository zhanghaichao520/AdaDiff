# 本地模型embedding
python amazon_text_emb.py \
  --mode local \
  --dataset Musical_Instruments \
  --model_name_or_path /home/peiyu/PEIYU/LLM_Models/Qwen/Qwen3-Embedding-8B \
  --batch_size 128 \
  --pca_dim 512 

# API模型embedding
python amazon_text_emb.py \
  --mode api \
  --dataset Toys_and_Games \
  --sent_emb_model text-embedding-3-large \
  --openai_api_key sk-WR6aJpZ81gACet0aN8wXYx0lUfC0WgjjcrFRZAJKatkeuTke \
  --openai_base_url https://yunwu.ai/v1 \
  --batch_size 256 \
  --pca_dim 512
