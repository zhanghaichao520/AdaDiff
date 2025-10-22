#!/bin/bash

# MovieLens 图片下载脚本
# 参考 Amazon 的处理方式

# 数据集名称
DATASET="ml-1m"

# 下载电影海报
python load_movielens_figures.py \
    --dataset $DATASET \
    --save_root ../datasets

echo "MovieLens 图片下载完成！"