# 🧩 GenRec-Factory 数据处理与Embedding

本项目提供从 **原始数据下载 → 数据预处理 → 文本与图像 Embedding 生成 → 多模态融合** 的一站式处理脚本。  
以 Amazon 与 MovieLens 为例。


## 📦 1. 下载数据集

从公开源下载 Amazon 或 MovieLens 数据集：

```bash
# Amazon 数据集
python download_data.py --source amazon --dataset Sports_and_Outdoors

# MovieLens 数据集
python download_data.py --source movielens --dataset ml-1m
```


## 🖼️ 2. 下载图片资源

若数据包含图像内容，可运行以下命令下载对应图片：

```bash
# Amazon 类数据集
python download_images.py --dataset_type amazon --dataset Sports_and_Outdoors

# MovieLens 数据集
python download_images.py --dataset_type movielens --dataset ml-1m
```



## 🧹 3. 数据预处理

对原始数据执行清洗、格式化与标准化：

```bash
# Amazon
python process_data.py --dataset_type amazon --dataset Sports_and_Outdoors

# MovieLens
python process_data.py --dataset_type movielens --dataset ml-1m
```

---

## 🔠 4. Embedding 生成

### 生成本地 T5 文本嵌入 (PCA 到 512d):

```bash
python process_embedding.py \
    --embedding_type text_local \
    --dataset Baby \
    --model_name_or_path sentence-transformers/sentence-t5-base \
    --pca_dim 512
```

### 生成 OpenAI API 文本嵌入:

```bash
python process_embedding.py \
    --embedding_type text_api \
    --dataset Baby \
    --sent_emb_model text-embedding-3-large \
    --pca_dim 512
```

### 生成 CLIP 图像嵌入:


```bash
python process_embedding.py \
    --embedding_type image_clip \
    --dataset Baby \
    --clip_model_name /home/wj/peiyu/LLM_Models/openai-mirror/clip-vit-base-patch32 \
    --pca_dim 512
```

### 生成 SASRec 协同嵌入:

```bash
python process_embedding.py \
    --embedding_type cf_sasrec \
    --dataset Baby \
    --sasrec_hidden_dim 64 \
    --sasrec_epochs 30 \
    --pca_dim 0
```

### 生成 Qwen-VL 融合嵌入:

```bash
python process_embedding.py \
    --embedding_type vlm_fused \
    --dataset Baby \
    --vlm_model_name_or_path Qwen/Qwen3-VL-7B-Instruct \
    --batch_size 16  # 注意调小 VLM batch size
    --pca_dim 512
```


## 5. 模态融合

```bash
python fusion_embedding.py \
    --dataset Baby \
    --text_model_tag "text-embedding-3-large" \
    --image_model_tag "clip-vit-base-patch32" \
    --fusion_epochs 10 \
    --batch_size 4096 \
    --fusion_out_dim 512
```