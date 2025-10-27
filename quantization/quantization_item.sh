python main.py \
  --model_name pq \
  --dataset_name Beauty \
  --embedding_modality text \
  --embedding_model text-embedding-3-large

# 多模态
python main.py \
  --model_name rqvae \
  --dataset_name Baby \
  --embedding_modality fused \
  --embedding_model clip-vit-base-patch32-forge512