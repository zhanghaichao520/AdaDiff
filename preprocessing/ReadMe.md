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

### 📘 文本特征

#### （1）使用本地模型

```bash
python generate_embeddings/text_embedding.py \
  --dataset_type amazon \
  --mode local \
  --dataset Toys_and_Games \
  --model_name_or_path /home/peiyu/PEIYU/LLM_Models/Qwen/Qwen3-Embedding-8B \
  --batch_size 128 \
  --pca_dim 512
```

#### （2）使用 API 模型

```bash
python generate_embeddings/text_embedding.py \
  --dataset_type amazon \
  --mode api \
  --dataset Baby \
  --sent_emb_model text-embedding-3-large \
  --openai_api_key sk-492a02uVsAauNrYsP4YRW2pvAsELc20hoHJeUh2Sop3GiL3C \
  --openai_base_url https://yunwu.ai/v1 \
  --batch_size 256 \
  --pca_dim 512
```

同样适用于 MovieLens：

```bash
python generate_embeddings/text_embeddings.py \
  --dataset_type movielens \
  --mode api \
  --dataset ml-1m \
  --sent_emb_model text-embedding-3-large \
  --openai_api_key sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --openai_base_url https://yunwu.ai/v1 \
  --batch_size 256 \
  --pca_dim 512
```

---

### 🖼️ 图像特征

使用 CLIP 模型提取视觉 Embedding：

```bash
python generate_embeddings/image_embedding.py \
  --dataset Baby \
  --model_name_or_path /home/wj/peiyu/LLM_Models/openai-mirror/clip-vit-base-patch32
```

### VL融合

 图像文本一起输入到VLLM

```bash
python generate_embeddings/vl_embedding.py \
  --dataset Baby \
  --vlm_model_name_or_path /home/wj/peiyu/LLM_Models/Qwen/Qwen3-VL-32B-Instruct \
  --batch_size 64 \
  --export_tag qwen7b \
  --pca_dim 512
```
### 协同特征

```bash
python generate_embeddings/cf_embedding.py --dataset Musical_Instruments --epochs 50 --hidden_dim 512
```

### CLIP文本+视觉

```bash
python generate_embeddings/clip_embedding.py \
  --dataset Baby \
  --dataset_type amazon \
  --model_name_or_path /home/wj/peiyu/LLM_Models/openai-mirror/clip-vit-base-patch32 
```

---

## 🔗 5. 多模态融合 (文本 + 视觉)

将文本与图像 Embedding 融合，生成最终的多模态表示：

```bash
python generate_embeddings/fuse_embedding.py \
  --dataset Baby \
  --text_model_tag "text-embedding-3-large" \
  --image_model_tag "clip-vit-base-patch32"
```

融合结果将保存至：

```
data/Musical_Instruments/embeddings/fused_emb.npy
```

---

## 📁 输出结构说明

处理完毕后，文件目录一般如下：

```
data/
├── Musical_Instruments/
│   ├── raw/                     # 原始数据
│   ├── processed/               # 清洗后的数据
│   ├── images/                  # 下载的图像
│   ├── embeddings/
│   │   ├── text_emb.npy         # 文本 embedding
│   │   ├── image_emb.npy        # 图像 embedding
│   │   └── fused_emb.npy        # 多模态融合 embedding
│   └── meta.json                # 元数据
└── ml-1m/
    ├── ...
```

---

## 🧠 提示

* 若使用本地模型，请确保路径正确且 GPU 可用；
* 若使用 API 模式，请提前配置好 `--openai_api_key` 与代理地址；
* 若希望加速 PCA，可使用 `--pca_dim 512` 参数压缩维度。

---

## 📜 作者与引用

本流程改编自 **GenRec-Factory** 预处理模块，适用于多模态生成式推荐数据准备阶段。

```

---

是否希望我帮你在这个 README 中再补上一个「🧩 接下来的步骤」章节，比如：
- 如何输入到 RQ-VAE；
- 如何生成 codebook；
- 如何将 embedding 转成 token（用于 MMGRec / TIGER）？
```
