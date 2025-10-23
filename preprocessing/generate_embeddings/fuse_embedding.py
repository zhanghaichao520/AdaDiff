import os
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA # <-- 1. 导入 PCA
import sys

def load_embedding(path):
    """加载 .npy 嵌入文件"""
    if not os.path.exists(path):
        print(f"错误: 自动查找失败，找不到文件 {path}")
        return None
    print(f"成功找到并加载: {path}")
    return np.load(path)

def main(args):
    # 1. 自动构建输入路径
    embeddings_dir = os.path.join(args.data_root, args.dataset, "embeddings")
    
    text_emb_file = f"{args.dataset}.emb-text-{args.text_model_tag}.npy"
    text_emb_path = os.path.join(embeddings_dir, text_emb_file)
    
    image_emb_file = f"{args.dataset}.emb-image-{args.image_model_tag}.npy"
    image_emb_path = os.path.join(embeddings_dir, image_emb_file)

    print(f"正在查找文本嵌入: {text_emb_path}")
    print(f"正在查找图像嵌入: {image_emb_path}")

    # 2. 加载嵌入
    text_emb = load_embedding(text_emb_path)
    image_emb = load_embedding(image_emb_path)

    if text_emb is None or image_emb is None:
        print("缺少必要的嵌入文件，请检查你的 --text_model_tag 和 --image_model_tag 是否正确。")
        sys.exit(1)

    # 3. 验证
    print(f"文本嵌入维度: {text_emb.shape}")
    print(f"图像嵌入维度: {image_emb.shape}")

    if text_emb.shape[0] != image_emb.shape[0]:
        print(f"错误: 物品数量不匹配! 文本 {text_emb.shape[0]} vs 图像 {image_emb.shape[0]}")
        sys.exit(1)

    # 4. L2 归一化 (除非被禁用)
    if not args.no_normalize:
        print("正在对文本和图像嵌入进行 L2 归一化...")
        text_emb = normalize(text_emb, norm='l2', axis=1)
        image_emb = normalize(image_emb, norm='l2', axis=1)
        print("归一化完成。")
    else:
        print("跳过 L2 归一化。")

    # 5. 拼接
    print("正在拼接嵌入...")
    fused_emb = np.concatenate([text_emb, image_emb], axis=1)
    print(f"拼接完成。中间维度: {fused_emb.shape}") # e.g., (N, 1024)

    # ==========================================================
    # --- 6. (新增) 对融合后的向量进行 PCA 降维 ---
    # ==========================================================
    if args.pca_dim > 0:
        print(f"\n对融合后的嵌入应用 PCA 降维，目标维度: {args.pca_dim}")
        if fused_emb.shape[1] < args.pca_dim:
            print(f"原始维度 ({fused_emb.shape[1]}) 小于目标维度 ({args.pca_dim})，跳过降维。")
        else:
            pca = PCA(n_components=args.pca_dim)
            fused_emb = pca.fit_transform(fused_emb)
            print(f"降维后最终维度: {fused_emb.shape}，保留方差: {sum(pca.explained_variance_ratio_):.4f}")
    else:
        print("pca_dim <= 0，跳过最终的 PCA 降维。")
    # ==========================================================

    # 7. 自动构建输出路径
    fused_tag = f"text_{args.text_model_tag}-image_{args.image_model_tag}"
    fused_filename = f"{args.dataset}.emb-multimodal-{fused_tag}.npy"
    output_path = os.path.join(embeddings_dir, fused_filename)

    # 8. 保存
    print(f"正在保存融合后的嵌入到: {output_path}")
    np.save(output_path, fused_emb)
    print("🎉 融合完成！")

def parse_args():
    parser = argparse.ArgumentParser(description="自动查找、融合文本和图像嵌入向量。")
    
    # --- 关键输入 ---
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称 (例如: Musical_Instruments, Home, ml-1m)')
    parser.add_argument('--text_model_tag', type=str, required=True,
                        help='文本模型的标签 (例如: "text-embedding-3-large")')
    parser.add_argument('--image_model_tag', type=str, required=True,
                        help='图像模型的标签 (例如: "clip-vit-base-patch32")')
    
    # --- 可选配置 ---
    parser.add_argument('--data_root', type=str, default="../datasets",
                        help='数据集的根目录')
    parser.add_argument('--no_normalize', action='store_true',
                        help='(可选) 禁用 L2 归一化，直接进行拼接')
    
    # --- 2. 增加 pca_dim 参数 ---
    parser.add_argument('--pca_dim', type=int, default=512,
                        help='(可选) 对*融合后*的嵌入应用 PCA 降维到此维度。<= 0 表示不降维。')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)