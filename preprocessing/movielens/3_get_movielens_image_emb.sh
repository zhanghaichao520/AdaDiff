#!/bin/bash

# MovieLens 图片特征提取脚本
# 参考 Amazon 的处理方式

export CUDA_VISIBLE_DEVICES=1

# 数据集名称
DATASET="ml-1m"

# 使用 CLIP 模型提取图片特征
python movielens_image_emb.py \
    --dataset $DATASET \
    --save_root ../datasets \
    --backbone ViT-L/14 \
    --model_cache_dir ../cache_models/clip

echo "MovieLens 图片特征提取完成！"
