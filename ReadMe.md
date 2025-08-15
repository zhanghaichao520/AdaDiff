# Multimodal RQ-VAE

# 快速跑通整个流程

### Data process

指定对应的数据集名称，会实现下载，预处理，embedding的过程

```
cd dataloader

# 整体一起运行完
bash run_pipeline.sh --dataset Baby

# 分步骤运行
# Step 1: Download raw data
bash 0_download_dataset.sh

# 目前不需要这一步，我们先聚焦于单模态
# Step 2: Download images
# bash 1_load_figure.sh

# Step 3: Process interaction data
bash 2_process.sh

# Step 4: Generate text embeddings
bash 3_get_text_emb.sh

# 目前不需要这一步，我们先聚焦于单模态
# Step 5: Generate image embeddings
bash 4_get_image_emb.sh

# 目前不需要这一步，我们先聚焦于单模态
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

**不需要动的模块**

`main.py`: 控制整体的训练流程, 训练流程是固定的，数据的输入的固定的

`trainer.py`: 控制模型的训练策略，输入和输出都是确定的，只需要改模型内部的结构

`dataset.py`: 数据的处理，加载是固定的，一套策略

**需要动的模块**

`evaluate.py`: 可以增加不同的评估方式

`/models`文件夹里面包括所有不同的模型，比如vq, rqvae,要新加模型，就可以新建一个模型名称，然后把所有的都放进去

`utils.py`: 根据需要更新，如果某个方法太通用或太融于，可以放进来

### Recommendation

**不需要动的模块**

`pipeline.py`: 控制整体的训练流程, 训练流程是固定的，数据的输入的固定的

`trainer.py`: 控制模型的训练策略，输入和输出都是确定的，只需要改模型内部的结构

`evaluator`: 评估脚本，所有的都最后转换为id进行评估

`dataset.py`: 数据的处理，加载是固定的，一套策略

**需要动的模块**

`/models/`: 里面每一个文件夹代表一个新的模型，比如encoder_decoder, encoder_retrieve， 增加新的模型，需要在对应文件夹增加model.py以及对应的工具类，input和output要和外部的接口统一

# 添加新量化模型的分步指南

---

假设我们要添加一个名为 **`VQVAE`** 的新模型。

##### 步骤 1：创建模型文件

在 `models/` 目录下，为您的新模型创建一个 Python 文件。

* **约定**：文件名必须是模型名称的**小写**形式。
* **示例**：为 `VQVAE` 模型创建文件 `models/vqvae.py`。

文件结构如下：
```
/quantization/
└── models/
    ├── abstract_vq.py
    ├── __init__.py
    ├── rqvae.py
    └── vqvae.py         <-- 新建这个文件
```

##### 步骤 2：实现模型类

在您新建的 `models/vqvae.py` 文件中，定义您的模型类。

* **约定 1 (继承)**：您的模型类必须继承自 `AbstractVQ`，以确保它拥有标准的接口。
    ```python
    from .abstract_vq import AbstractVQ
    ```
* **约定 2 (命名)**：您的类名必须是模型名称的**大写**形式（或驼峰式，但推荐统一为大写以匹配 `RQVAE`）。
    ```python
    class VQVAE(AbstractVQ):
        # ... 你的代码 ...
    ```
* **约定 3 (实现接口)**：您必须实现 `AbstractVQ` 中定义的几个核心方法：
    * `__init__(self, config: dict, input_size: int)`: 构造函数，接收从 `.yaml` 文件加载的配置和输入向量的维度。
    * `forward(self, xs: torch.Tensor)`: 模型的前向传播，**必须**返回一个元组 `(reconstruction, loss, codes)`。
    * `get_codes(self, xs: torch.Tensor)`: 用于推理，只返回离散码 `codes`。
    * `compute_loss(self, forward_outputs, original_xs)`: 计算总损失，**必须**返回一个包含 `'loss_total'` 键的字典。`Trainer` 将使用这个返回值进行反向传播。

**示例 (`models/vqvae.py`):**
```python
from .abstract_vq import AbstractVQ
import torch
# ... 其他导入 ...

class VQVAE(AbstractVQ):
    def __init__(self, config: dict, input_size: int):
        super().__init__(config)
        # 从 config['vqvae']['model_params'] 中解析参数并构建模型...
        
    def forward(self, xs):
        # 实现前向逻辑...
        return reconstruction, vq_loss, codes
        
    def get_codes(self, xs):
        # 实现获取codes的逻辑...
        return codes
        
    def compute_loss(self, forward_outputs, original_xs):
        # 实现计算总损失的逻辑...
        return {"loss_total": total_loss, "recon_loss": recon_loss}
```

##### 步骤 3：创建配置文件

在 `configs/` 目录下，为您的新模型创建一个 `.yaml` 配置文件。

* **约定**：配置文件名必须遵循 `{model_name}_config.yaml` 的格式。
* **示例**：为 `VQVAE` 模型创建文件 `configs/vqvae_config.yaml`。

**示例 (`configs/vqvae_config.yaml`):**
```yaml
# 通用设置
common:
  device: 'cuda:0'

# VQVAE 模型专属配置
vqvae:
  model_params:
    # 在这里定义 VQVAE 需要的架构参数
    hidden_size: 512
    num_embeddings: 4096
    embedding_dim: 256
  
  training_params:
    # 在这里定义 VQVAE 训练时需要的参数
    batch_size: 512
    epochs: 1000
    lr: 0.001
    optimizer: "AdamW"
    commitment_cost: 0.25
```

##### 如何运行您的新模型

现在，您可以直接通过 `main.py` 的命令行参数来运行您的新模型：

```bash
python main.py \
    --model_name=vqvae \
    --dataset_name=YourDatasetName \
    --embedding_suffix=your_suffix
```
我们的通用 `main.py` 脚本会自动完成以下工作：
1.  根据 `--model_name=vqvae` 找到并加载 `configs/vqvae_config.yaml`。
2.  根据 `--model_name=vqvae` 找到并加载 `models/vqvae.py` 文件中的 `VQVAE` 类。
3.  将配置传递给模型和通用 `Trainer`，开始整个训练和码本生成流程。