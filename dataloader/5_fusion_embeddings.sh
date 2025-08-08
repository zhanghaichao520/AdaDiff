# 数据集名称统一在这里改
DATASET="Baby"

# 直接拼接
python fusion_embeddings.py \
    --method concat \
    --dataset_name $DATASET \
    --embed_dim 512

# CLIP对齐
python fusion_embeddings.py \
    --method clip-align \
    --dataset_name $DATASET \
    --epochs 15 \
    --lr 1e-4 \
    --batch_size 256 \
    --embed_dim 512

# 投影融合
python fusion_embeddings.py \
    --method projection \
    --dataset_name $DATASET \
    --epochs 10 \
    --embed_dim 512

# 交叉注意力融合
python fusion_embeddings.py \
    --method cross-attention \
    --dataset_name $DATASET \
    --epochs 20 \
    --embed_dim 512 \
    --nhead 4
