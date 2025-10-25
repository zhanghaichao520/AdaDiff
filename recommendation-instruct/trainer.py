# 文件路径: instruction_peft_trainer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import logging
import argparse
import os
from pathlib import Path
import math
import pprint # 用于打印配置

# 假设 instruction_dataset.py 在同一目录下或可导入
try:
    from dataset import InstructionDataset, DataCollatorForInstructionTuning
except ImportError:
    print("错误：无法导入 instruction_dataset.py")
    raise

# ✅ 导入新的工具函数
try:
    from utils import load_instruction_config, setup_instruction_logging
except ImportError:
    print("错误：无法导入 instruction_utils.py")
    raise

# 使用 root logger (由 setup_instruction_logging 配置)
logger = logging.getLogger(__name__) # 获取当前模块的 logger

# ============================================
#            训练与评估逻辑 (保持不变)
# ============================================
def train_instruction_epoch(model, train_loader, optimizer, scheduler, device):
    """执行一个指令微调训练周期"""
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        # DataCollator 已将数据转为 Tensor，只需移动设备
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # 模型 forward (PEFT 模型会自动处理)
        # 对于 CausalLM，Hugging Face 模型会自动计算损失
        outputs = model(**batch)
        loss = outputs.loss
        
        if loss is None:
            logger.warning("模型未返回损失，跳过此批次。请检查模型配置和 DataCollator。")
            continue
            
        loss.backward()
        optimizer.step()
        scheduler.step() # 通常学习率调度器在每个 step 更新
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate_loss(model, eval_loader, device) -> float:
    """在评估集上简单计算损失 (作为 Perplexity 的代理)"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0 # 用于更精确地计算 Perplexity

    for batch in tqdm(eval_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss

        if loss is not None:
             # 计算批次中用于计算损失的 token 数量
             valid_labels = batch['labels'] != -100
             num_valid_tokens = valid_labels.sum().item()
             # 累加加权损失 (loss * num_tokens)
             total_loss += loss.item() * num_valid_tokens
             total_tokens += num_valid_tokens
        else:
             logger.warning("评估时模型未返回损失。")

    if total_tokens == 0:
        logger.warning("评估集上没有有效的 token 用于计算损失。")
        return float('inf')
        
    avg_loss = total_loss / total_tokens
    # 可以选择返回 avg_loss 或 perplexity
    # perplexity = math.exp(avg_loss)
    # return perplexity
    return avg_loss # 返回平均损失更直接

# ============================================
#                主程序入口 (已修改)
# ============================================

def main():
    parser = argparse.ArgumentParser(description="基于 PEFT (LoRA) 的智能指令微调训练脚本")

    # --- ✅ 简化命令行参数 ---
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (e.g., Baby)')
    parser.add_argument('--base_model_alias', type=str, required=True, help='配置文件中定义的模型别名 (e.g., gpt2-medium)')

    # --- 保留关键的可覆盖超参数 (设为 None 表示默认使用配置文件) ---
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数 (覆盖配置)')
    parser.add_argument('--batch_size', type=int, default=None, help='训练批量大小 (覆盖配置)')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='评估批量大小 (覆盖配置)')
    parser.add_argument('--lr', type=float, default=None, help='学习率 (覆盖配置)')
    parser.add_argument('--max_seq_len', type=int, default=None, help='最大序列长度 (覆盖配置)')
    parser.add_argument('--lora_r', type=int, default=None, help='LoRA 的秩 (覆盖配置)')
    parser.add_argument('--lora_alpha', type=int, default=None, help='LoRA 的 alpha (覆盖配置)')
    parser.add_argument('--device', type=str, default=None, help='训练设备 (覆盖配置, e.g., cuda:1)')
    # 你可以根据需要添加更多可覆盖的参数

    args = parser.parse_args()

    # === 🚀 核心改动：使用工具函数加载和处理配置 ===
    try:
        config = load_instruction_config(args.dataset_name, args.base_model_alias, args)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"加载或处理配置失败: {e}")
        return # 终止程序

    # === 设置日志 (使用 config 中推导出的路径) ===
    setup_instruction_logging(config['log_path'])

    # 打印最终使用的配置
    logger.info("=" * 30 + " 最终配置 " + "=" * 30)
    config_str = pprint.pformat(config)
    logger.info("\n" + config_str)
    logger.info("=" * (60 + len(" 最终配置 ")))

    # --- 后续流程使用 config 中的值 ---
    device = torch.device(config['device'] if torch.cuda.is_available() and 'cuda' in config['device'] else "cpu")
    logger.info(f"使用设备: {device}")

    # 1. 加载 Tokenizer (使用 config 中的路径)
    tokenizer_path = config['token_params']['tokenizer_path']
    logger.info(f"加载 Tokenizer 从: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            cache_dir=config['paths'].get('model_cache_dir') # 使用缓存路径
        )
    except Exception as e:
        logger.error(f"加载 Tokenizer 失败: {e}")
        return

    # 2. 加载基础模型 (使用 config 中的路径)
    base_model_path = config['base_model_path']
    logger.info(f"加载基础模型从: {base_model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            cache_dir=config['paths'].get('model_cache_dir') # 使用缓存路径
            # torch_dtype=torch.bfloat16
        ).to(device)
    except Exception as e:
        logger.error(f"加载基础模型失败: {e}")
        return

    # 3. 配置并应用 PEFT (使用 config 中的 peft_params)
    logger.info("配置 PEFT (LoRA)...")
    peft_cfg = config['peft_params']
    lora_config = LoraConfig(
        r=peft_cfg['lora_r'],
        lora_alpha=peft_cfg['lora_alpha'],
        lora_dropout=peft_cfg['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=peft_cfg.get('target_modules') # 从配置读取 target_modules
    )
    logger.info("应用 PEFT 到基础模型...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 创建数据集和数据加载器 (使用 config 中的路径和参数)
    logger.info("加载数据集...")
    try:
        train_dataset = InstructionDataset(config['train_jsonl'], tokenizer)
        valid_dataset = InstructionDataset(config['valid_jsonl'], tokenizer)
    except FileNotFoundError:
        logger.error("训练或验证数据文件未找到，请检查路径。")
        return

    data_collator = DataCollatorForInstructionTuning(tokenizer, max_length=config['token_params']['max_seq_len'])

    train_cfg = config['training_params']
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        collate_fn=data_collator,
        shuffle=True,
        num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=train_cfg['eval_batch_size'],
        collate_fn=data_collator,
        shuffle=False,
        num_workers=4
    )

    # 5. 初始化优化器和学习率调度器 (使用 config 中的参数)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay']
    )

    num_training_steps = len(train_loader) * train_cfg['epochs'] // train_cfg['gradient_accumulation_steps']
    num_warmup_steps = int(num_training_steps * train_cfg['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # 6. 训练循环 (逻辑不变, 使用 config 中的参数)
    best_val_loss = float('inf')
    output_dir = config['output_dir'] # 使用 config 中的输出路径

    logger.info("开始指令微调...")
    for epoch in range(1, train_cfg['epochs'] + 1):
        logger.info(f"--- Epoch {epoch}/{train_cfg['epochs']} ---")

        # (训练和评估函数调用不变)
        avg_train_loss = train_instruction_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Epoch {epoch} 训练完成. 平均训练损失: {avg_train_loss:.4f}")

        avg_val_loss = evaluate_loss(model, valid_loader, device)
        logger.info(f"Epoch {epoch} 评估完成. 平均验证损失: {avg_val_loss:.4f}")

        # 保存最佳模型 (逻辑不变)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"🚀 新的最佳验证损失: {best_val_loss:.4f}。保存 PEFT adapter 到 {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    logger.info("指令微调完成！")
    logger.info(f"最佳 PEFT adapter 保存在: {output_dir}")

if __name__ == "__main__":
    main()