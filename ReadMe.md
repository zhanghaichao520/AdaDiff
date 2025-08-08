# Multimodal RQ-VAE

### Data process

```
cd dataloader

# 整体一起运行完
bash run_pipeline.sh --dataset Beauty

# 分步骤运行
# Step 1: Download raw data
bash 0_download_dataset.sh

# Step 2: Download images
bash 1_load_figure.sh

# Step 3: Process interaction data
bash 2_process.sh

# Step 4: Generate text embeddings
bash 3_get_text_emb.sh

# Step 5: Generate image embeddings
bash 4_get_image_emb.sh

```