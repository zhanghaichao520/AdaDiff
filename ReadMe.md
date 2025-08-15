# Multimodal RQ-VAE

# 训练流程

### Data process

```
cd dataloader

# 整体一起运行完
bash run_pipeline.sh --dataset Baby

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

# Step 6: Generate fusion embeddings
bash 5_fusion_embeddings.sh

```

### Quantization

```
cd quantization

# 多模态
python main.py --quantizer_name rqvae --dataset_name Beauty --embedding_suffix "fused-concat"

# 纯文本
python main.py --quantizer_name rqvae --dataset_name Musical_Instruments --embedding_suffix "td"
```

### Train Recommendation

```
cd recommendation

python main.py --category=Baby --model=encoder_decoder

python main.py --category=Musical_Instruments --model=encoder_decoder
```


# 代码规范

### Quantization

`main.py`控制所有的模型的训练

每一个模型，都有一个对应名称的文件，里面包括了当前模型需要的所有模块，比如rqvae就是vq+mlp的文件，此外，还需要配置一个train.py用于训练。如果需要扩展新的模型，新建一个当前模型的文件夹。在`main.py`里面增加切换的逻辑

`/rqvae`文件夹里面包括一个模型的所有模块，包括vq、mlp、loss等

### Recommendation

**不需要动的模块**

`pipeline.py`: 控制整体的训练流程, 训练流程是固定的，数据的输入的固定的

`trainer.py`: 控制模型的训练策略，输入和输出都是确定的，只需要改模型内部的结构

`evaluator`: 评估脚本，所有的都最后转换为id进行评估

`dataset.py`: 数据的处理，加载是固定的，一套策略

**需要动的模块**

`/models/`: 里面每一个文件夹代表一个新的模型，比如encoder_decoder, encoder_retrieve， 增加新的模型，需要在对应文件夹增加model.py以及对应的工具类，input和output要和外部的接口统一