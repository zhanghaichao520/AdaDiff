import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional

IGNORE_INDEX = -100 # PyTorch CrossEntropyLoss 默认忽略的索引

class InstructionDataset(Dataset):
    """
    处理 Llama Factory 格式 JSONL 数据的 Dataset。
    格式: {"instruction": "...", "input": "...", "output": "..."}
    """
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer):
        super(InstructionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"警告：跳过无法解析的行: {line.strip()}")
        except FileNotFoundError:
            print(f"错误：找不到数据文件 {data_path}")
            raise

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, str]:
        # 直接返回原始的字典
        return self.data[index]

def format_prompt(example: Dict[str, str]) -> str:
    """
    根据 instruction, input, output 构建最终的训练序列文本。
    这是一个简单的模板，你可以根据需要修改。
    """
    if example.get("input"):
        return (
            f"<s>[INST] {example['instruction']} \n{example['input']} [/INST]\n"
            f"{example['output']} </s>"
        )
    else:
        return (
            f"<s>[INST] {example['instruction']} [/INST]\n"
            f"{example['output']} </s>"
        )

def format_prompt_without_output(example: Dict[str, str]) -> str:
    """仅构建 Prompt 部分，用于计算 labels 时确定掩码范围"""
    if example.get("input"):
        return f"<s>[INST] {example['instruction']} \n{example['input']} [/INST]\n"
    else:
        return f"<s>[INST] {example['instruction']} [/INST]\n"


class DataCollatorForInstructionTuning:
    """
    数据整理器 (已修正)。
    核心：手动进行 Tokenize 和 Padding，确保长度一致后再转 Tensor。
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("Tokenizer 没有 pad token，将使用 eos token 作为 pad token。")
            # 确保 pad_token_id 也被设置
            if tokenizer.pad_token_id is None:
                 tokenizer.pad_token_id = tokenizer.eos_token_id
                 logger.warning(f"设置 pad_token_id = {tokenizer.eos_token_id}")

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        # 1. 构建文本 (不变)
        full_texts = [format_prompt(f) for f in features]
        prompt_texts = [format_prompt_without_output(f) for f in features]

        # 2. ✅ Tokenize 完整序列 (不返回 Tensor)
        # 这会返回 List[List[int]]
        full_tokens_list = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            truncation=True,
            # return_tensors=None # 确保不返回 Tensor
        )["input_ids"]

        # 3. ✅ Tokenize Prompt 序列 (不返回 Tensor, 仅获取长度)
        prompt_tokens_list = self.tokenizer(
            prompt_texts,
            max_length=self.max_length,
            truncation=True,
            # return_tensors=None, # 确保不返回 Tensor
            add_special_tokens=False # 通常 prompt 模板已包含
        )["input_ids"]
        prompt_lengths = [len(p) for p in prompt_tokens_list]

        # 4. ✅ 手动 Padding 完整序列
        # 使用 tokenizer.pad 方法，它可以正确处理 padding
        padded_full_inputs = self.tokenizer.pad(
            {"input_ids": full_tokens_list}, # 需要传入字典格式
            padding="max_length" if self.max_length else "longest",
            max_length=self.max_length,
            return_tensors="pt", # 在这里要求返回 Tensor
            return_attention_mask=True # 同时生成 attention_mask
        )

        # 5. ✅ 创建 Labels (基于手动 padding 后的 input_ids)
        input_ids = padded_full_inputs["input_ids"]
        labels = input_ids.clone()

        for i in range(len(labels)):
            prompt_len = prompt_lengths[i]
            # (掩码逻辑不变)
            # 确保 prompt_len 不超过序列实际长度 (padding 后)
            actual_seq_len = (padded_full_inputs["attention_mask"][i]).sum().item()
            mask_len = min(prompt_len, actual_seq_len) # 防止越界
            labels[i, :mask_len] = IGNORE_INDEX

        # (Padding Token 掩码不变)
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        # 6. ✅ 构建最终输出字典
        final_batch = {
            "input_ids": input_ids,
            "attention_mask": padded_full_inputs["attention_mask"],
            "labels": labels
        }

        return final_batch