# GenRec
<p align="center">
  <img src="./asset/Logo.png" width="200">
</p>

$\textbf{GenRec}$: A unified **Gen**erative **Rec**ommendation toolbox that simplifies end-to-end generative recommender research.  
📘 [arXiv Paper (coming soon)](https://arxiv.org/abs/2501.xxxxx)  
:point_right: Check our **survey on generative recommendation (2025)** (to appear).  
:point_right: Explore **awesome resources on GenRec** → [Generative Recommendation Resources](https://github.com/yourname/GenRecHub-Resources).  

---

## 🔧 Toolbox Overview
<p align="center">
  <img src="./images/GenRecHub.png" width="500">
</p>

GenRecHub provides a modular and reproducible pipeline for **end-to-end generative recommendation**.
It unifies data preprocessing, tokenization, generation, and evaluation under a single configuration-driven framework.

---

# ⚙️ Capability Matrix

## Dataset

| Dataset | Implemented |
|--------------------|--------------|
| Amazon | ✅ |

## 💬 Embedding Extraction
| Category | Component / Method | Done |
|-----------|--------------------|--------------|
| 🧠 Textual | Sentence Embedding | ✅ |
|  | OpenAI Embedding API | ✅ |
|  | Local LLM Embedding (Qwen2.5, MiniCPM) | ✅ |
| 🖼️ Visual | CLIP / BLIP2 Encoder | ✅ |
|  | Multimodal Fusion | ✅ |
| 👥 Collaborative | SASRec Sequence Embedding | ✅ |
| 🧩 Management | PCA Compression & Storage | ✅ |

## 🧩 Quantization
| Category | Component / Method | Done |
|-----------|--------------------|--------------|
| 🔸 Residual Family | RQ-VAE | ✅ |
|  | R-KMeans | ✅ |
|  | VQ-VAE   | ✅ |
|  | R-VQ     | ✅ |
| 🔹 Product Family | OPQ | ✅ |
|  | PQ   | ✅ |

## ⚙️ Recommendation Architecture
| Category | Component / Method | Done |
|-----------|--------------------|--------------|
| 🧠 Encoder–Decoder | TIGER | ✅ |
| 💬 Decoder-Only | GPT2 | ✅ |
|                 | LLM(Qwen, LLaMA) | ✅ |
| 🔍 Encoder-Retrieval | RPG | ✅ |
| 🔧 Plugins | Beam Search | - |
|  | Prefix Tree Constraint | - |








---

## 📦 Supported Models

| **Category** | **Model** | **Paper** | **Conference/Journal** | **Code** | **Done** |
|---------------|------------|-----------|------------------------|-----------|-----------|
| **Encoder-Decoder** | TIGER | [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) | NIPS' 23 | rqvae.py + TIGER.py | ✅ |
| **Encoder-Retrieval** | RPG | [Generating Long Semantic IDs in Parallel for Recommendation](http://arxiv.org/abs/2506.05781) | KDD' 25 | opq.py + RPG.py | ✅ | 
| **Quantization** | LETTER | [Learnable Item Tokenization for Generative Recommendation](https://dl.acm.org/doi/10.1145/3627673.3679569) | CIKM' 24 | opq.py + RPG.py | - |



