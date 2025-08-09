import os, argparse, numpy as np, torch

def scan_inter(inter_path, limit=100000):
    used = set()
    with open(inter_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for i, line in enumerate(f):
            if not line.strip(): continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) != 3: continue
            uid, seq_str, tgt_str = cols
            if seq_str:
                for s in seq_str.split(' '):
                    if s: used.add(int(s))
            if tgt_str:
                used.add(int(tgt_str))
            if limit and i >= limit: break
    return used

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', required=True)      # ../datasets/Baby/Baby.emb-fused-*.npy
    ap.add_argument('--inter', required=True)    # ../datasets/Baby/training/Baby.train.inter
    ap.add_argument('--pt', required=True)       # ../datasets/Baby/codebook.pt
    args = ap.parse_args()

    E = np.load(args.emb)
    N = E.shape[0]
    print(f'[embeddings] shape={E.shape} -> N={N}')

    used = scan_inter(args.inter)
    mn, mx = (min(used), max(used)) if used else (None, None)
    print(f'[inter] min_id={mn}, max_id={mx}, count_distinct={len(used)}')
    bad = [i for i in used if i < 0 or i >= N]
    if bad:
        print(f'❌ 有越界 item_id（示例前10个）：{bad[:10]}')
    else:
        print('✅ 所有 item_id 都在 [0, N-1] 范围内')

    if mn == 1 and mx == N:  # 常见“从1开始”的错位信号
        print('⚠️ 观测到 item_id 可能从 1 开始。如果 embeddings 第0行不是 PAD，索引会错位。')

    codebook = torch.load(args.pt, map_location='cpu')
    print(f'[codebook.pt] shape={tuple(codebook.shape)}, dtype={codebook.dtype}')
    assert codebook.dtype in (torch.int64, torch.long), 'codebook 需要是 int64/long'
    H, Nc = codebook.shape
    if Nc != N:
        print(f'❌ codebook 列数 N={Nc} 与 embeddings 行数 N={N} 不一致')
    else:
        print('✅ codebook.shape[1] 与 embeddings.shape[0] 一致')
    # 抽样几条做索引尝试
    sample_ids = [0, 1, mx//2 if mx else 0, mx or 0]
    sample_ids = [i for i in sample_ids if 0 <= i < N]
    if sample_ids:
        try:
            _ = codebook[:, sample_ids]
            print(f'✅ 抽样索引通过：{sample_ids}')
        except Exception as e:
            print(f'❌ 抽样索引失败：{e}')

if __name__ == '__main__':
    main()
