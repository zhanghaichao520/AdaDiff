python main.py \
  --model_name vqvae \
  --dataset_name Sports_and_Outdoors \
  --embedding_modality text \
  --embedding_model text-embedding-3-large

# 多模态
python main.py \
  --model_name rqvae \
  --dataset_name Baby \
  --embedding_modality fuse \
  --embedding_model clip-vit-base-patch32