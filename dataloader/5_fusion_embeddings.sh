# 直接拼接
python fusion_embeddings.py \
    --method concat \
    --dataset_name Beauty

# CLIP对齐
python fusion_embeddings.py \
    --method clip-align \
    --dataset_name Beauty \
    --epochs 15 \
    --lr 1e-4 \
    --batch_size 256 \
    --embed_dim 512

# 投影融合
python fusion_embeddings.py \
    --method projection \
    --dataset_name Beauty \
    --epochs 10 \
    --embed_dim 512

# 交叉注意力融合
python fusion_embeddings.py \
    --method cross-attention \
    --dataset_name Beauty \
    --epochs 20 \
    --embed_dim 512 \
    --nhead 4