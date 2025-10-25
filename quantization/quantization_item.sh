python main.py \
  --model_name mqvae \
  --dataset_name Toys_and_Games \
  --embedding_modality text \
  --embedding_model text-embedding-3-large

# 多模态
python main.py \
  --model_name rqvae \
  --dataset_name Baby \
  --embedding_modality fuse \
  --embedding_model clip-vit-base-patch32