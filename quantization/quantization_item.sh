python main.py \
  --model_name rkmeans \
  --dataset_name Toys_and_Games \
  --embedding_modality text \
  --embedding_model text-embedding-3-large

# 多模态
python main.py \
  --model_name rkmeans \
  --dataset_name Musical_Instruments \
  --embedding_modality multimodal \
  --embedding_model text_text-embedding-3-large-image_clip-vit-base-patch32