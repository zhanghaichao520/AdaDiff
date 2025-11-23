# UniGenRec
<p align="center">
  <img src="./asset/logo.png" width="200">
</p>

**UniGenRec** — A unified, modular, configuration-driven **Generative Recommendation** toolbox.  
It provides an end-to-end reproducible pipeline covering **Representation → Tokenization → Modeling → Training → Inference**.

📘 arXiv Paper (coming soon)  


# 🔥 Introduction
Generative Recommendation is rapidly emerging as a new paradigm, shifting from **scoring/matching** → **generative modeling**. However, the current GenRec ecosystem is **highly fragmented**:
- **Representation & Tokenization** are inconsistent (RQ-VAE, VQ-VAE, OPQ, RKMeans, LETTER…)  
- **Backbones** vary widely (Encoder–Decoder, Decoder-only LLMs, Retrieval-Hybrids)  
- **Training & Inference** pipelines differ significantly (beam search, prefix-tree, guided decoding)

As a result: **models are not comparable, not extensible, and often not reproducible**.

# 🎯 Goal

UniGenRec provides a **single, configuration-driven, plug-and-play GenRec stack**, unifying  **Representation → Tokenization → Backbone → Training → Inference** to enable reproducible research and fair comparison across models.

- **A fully unified GenRec stack**
- **Modular and plug-and-play components**
- **Reproducible experiments with config-based control**
- **Fair comparison across GenRec models**
- **First open-source standardization of SID-based modeling**


# 🔧 Pipeline Overview

```
Raw Data
↓
Download + Preprocessing
↓
Embedding Generation (Text / Image / CF / VLM)
↓
Multimodal Fusion (optional)
↓
Quantization (RQ-VAE / OPQ / PQ / RKMeans)
↓
Generative Recommender (TIGER / RPG / LETTER / LLMs)
↓
Inference (Beam Search / Prefix-tree / Contrastive Rerank)
```


---

# 🧱 Capability Matrix

| Dimension | Category | Supported Components | Status |
|----------|----------|----------------------|--------|
| **Data** | Datasets | Amazon, MovieLens | ✓ |
|          | Input Formats | Raw IDs, Embeddings, Codebooks (SID) | ✓ |
| **Representation** | Text | Qwen, T5, OpenAI Embedding API | ✓ |
|                   | Vision | CLIP ViT | ✓ |
|                   | Collaborative | SASRec | ✓ |
|                   | Fusion | Concat, MLP Fusion | ✓ |
| **Tokenization / Quantization** | Residual Family | RQ-VAE, Residual KMeans, Residual-VQ | ✓ |
|                                | Product Family | OPQ, PQ | ✓ |
|                                | Other | VQ-VAE, Multi-Codebook (RPG-style) | ✓ |
| **Backbone** | Encoder–Decoder | TIGER-style architectures | ✓ |
|              | Decoder-only LLM | GPT-2, Qwen, LLaMA | ✓ |
|              | Retrieval-Hybrid | RPG-style architectures | ✓ |
| **Training** | Objectives | LM Loss, Contrastive Loss, Hybrid Loss | ✓ |
|              | Paradigms | SFT, Alignment, Multi-stage Training | ✓ |
| **Inference** | Decoding | Greedy, Beam Search | ✓ |
|               | Constraints | Prefix-Tree | ✓  |




# 🚀 Quick Start


**Requirements**
- Python **3.10** (recommended)
- PyTorch, CUDA, and other dependencies will be installed automatically via `requirements.txt`

```bash
git clone https://github.com/yourname/UniGenRec
cd UniGenRec
pip install -r requirements.txt
```
## 1 Data Preprocessing

For dataset downloading, cleaning, formatting, and multimodal data preparation  
(**including text/image extraction, interaction filtering, metadata normalization**),  
please refer to the dedicated guide:

👉 **See detailed tutorial:**  
[GenRec-Factory Data Processing & Embedding Guide](./preprocessing/ReadMe.md)

This includes:

## 2 Quantization

```bash
cd quantization

python main.py \
  --model_name rqvae \
  --dataset_name Musical_Instruments \
  --embedding_modality text \
  --embedding_model text-embedding-3-large
```

## 3 Generative Recommendation Models

```bash
python main.py \
  --model TIGER \
  --dataset Baby \
  --quant_method rqvae
```
