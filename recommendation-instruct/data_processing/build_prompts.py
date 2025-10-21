import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path

def build_instruction_data_topk(
    codebook_path,
    source_data_path,
    output_path,
    instruction_text,
    history_prefix,
    item_separator="||",
    k=10
):
    """
    构建带负采样的 Top-K 指令微调数据。
    每条样本包含：
      - 用户历史（history）
      - 候选 items（target + k-1 negatives）
      - 输出顺序为目标在首，其余为随机负样本
    """
    print("-" * 30)
    print(f"Processing source: {source_data_path}")

    if not codebook_path.exists():
        print(f"❌ Codebook 不存在: {codebook_path}")
        return False
    if not source_data_path.exists():
        print(f"⚠️ 源数据不存在: {source_data_path}")
        return False

    # 读取 codebook
    with open(codebook_path, 'r', encoding='utf-8') as f:
        codebook = json.load(f)
    all_item_ids = list(codebook.keys())
    print(f"✅ Codebook 加载完成，共 {len(all_item_ids)} 个 items")

    # 输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count, skip = 0, 0
    with open(source_data_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        lines = fin.readlines()
        for line in tqdm(lines, total=len(lines), desc=f"Processing {source_data_path.name}"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                history_ids = [str(x) for x in data.get("history", [])]
                target_id = str(data.get("target"))
                if target_id is None or target_id not in codebook:
                    skip += 1
                    continue

                # 构建历史
                history_tokens = [codebook[h] for h in history_ids if h in codebook]

                # ---- 负采样逻辑 ----
                neg_pool = list(set(all_item_ids) - set(history_ids) - {target_id})
                if len(neg_pool) < (k - 1):
                    continue
                neg_samples = random.sample(neg_pool, k - 1)
                candidates = [target_id] + neg_samples
                random.shuffle(candidates)
                candidate_tokens = [codebook[c] for c in candidates if c in codebook]

                # ---- 构建输入 ----
                input_text = (
                    f"{history_prefix}\n"
                    f"[{item_separator.join(history_tokens)}].\n"
                    f"Candidates: [{item_separator.join(candidate_tokens)}].\n"
                    f"Please select the top {k} most likely items."
                )

                # ---- 构建输出 ----
                # 输出：目标在第一位，其余为随机负样本
                other_neg = random.sample(neg_samples, len(neg_samples))
                output_tokens = [codebook[target_id]] + [codebook[n] for n in other_neg]
                output_str = item_separator.join(output_tokens)

                sample = {
                    "instruction": instruction_text,
                    "input": input_text,
                    "output": output_str
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1

            except Exception as e:
                skip += 1
                print(f"⚠️ 解析行时出错: {e}")
                continue

    print("-" * 30)
    print(f"✅ 完成: {count} 条样本写入 {output_path}")
    print(f"⚠️ 跳过 {skip} 条异常或无效数据")
    print("-" * 30)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 Top-K 负采样式推荐指令数据")

    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--quant_method', type=str, required=True, help='量化方法名称，如 rqvae')
    parser.add_argument('--data_base_path', type=str, default='../../datasets', help='数据根路径')
    parser.add_argument('--codebook_dir_name', type=str, default='codebooks', help='Codebook 子目录')
    parser.add_argument('--instruction', type=str,
                        default="You are a helpful recommendation assistant that works on codebook tokens.",
                        help='指令文本')
    parser.add_argument('--history_prefix', type=str,
                        default="Given the following purchase history of a user:",
                        help='历史序列前缀')
    parser.add_argument('--item_separator', type=str, default="||", help='项目分隔符')
    parser.add_argument('--k', type=int, default=10, help='Top-K 候选数量')

    args = parser.parse_args()

    dataset_path = Path(args.data_base_path) / args.dataset_name
    codebook_filename = f"{args.dataset_name}.{args.quant_method}.codebook.json"
    codebook_path = dataset_path / args.codebook_dir_name / codebook_filename

    output_dir = dataset_path / "prompts_topk"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'valid', 'test']
    for split in splits:
        source_file = dataset_path / f"{args.dataset_name}.{split}.jsonl"
        output_file = output_dir / f"{args.dataset_name}.{args.quant_method}.{split}.top{args.k}.jsonl"

        build_instruction_data_topk(
            codebook_path,
            source_file,
            output_file,
            args.instruction,
            args.history_prefix,
            args.item_separator,
            k=args.k
        )

    print(f"\n✅ 所有数据分割构建完成！输出位于: {output_dir}\n")
