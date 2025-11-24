python main.py \
  --model_name rkmeans \
  --dataset_name Musical_Instruments \
  --embedding_modality cf \
  --embedding_model sasrec

# 多模态
python main.py \
  --model_name mm_rqvae \
  --dataset_name Baby \
  --text_embedding_model text-embedding-3-large \
  --image_embedding_model clip-vit-base-patch32 