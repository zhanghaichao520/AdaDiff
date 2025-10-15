import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# --- 1. 模型定义 (和之前一样，无需修改) ---

class ClipAlignmentModel(nn.Module):
    """方案二：CLIP对齐模型。"""
    def __init__(self, text_dim, image_dim, embed_dim=1024):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, embed_dim)
        self.image_projection = nn.Linear(image_dim, embed_dim)
    
    def forward(self, text_features, image_features):
        text_embeds = self.text_projection(text_features)
        image_embeds = self.image_projection(image_features)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return text_embeds, image_embeds

class ProjectionFusionModel(nn.Module):
    """方案三：线性投影 + 融合模型。"""
    def __init__(self, text_dim, image_dim, embed_dim=1024, projection_dim=1024):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, projection_dim)
        self.image_projection = nn.Linear(image_dim, projection_dim)
        self.final_projection = nn.Linear(projection_dim * 2, embed_dim)
        
    def forward(self, text_features, image_features):
        text_proj = torch.relu(self.text_projection(text_features))
        image_proj = torch.relu(self.image_projection(image_features))
        concatenated = torch.cat([text_proj, image_proj], dim=-1)
        fused_embeds = self.final_projection(concatenated)
        return fused_embeds, fused_embeds

class CrossAttentionFusionModel(nn.Module):
    """方案四：交叉注意力融合模型。"""
    def __init__(self, text_dim, image_dim, embed_dim=1024, nhead=4):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by nhead ({nhead})")
        self.text_projection = nn.Linear(text_dim, embed_dim)
        self.image_projection = nn.Linear(image_dim, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.final_fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_features, image_features):
        text_proj = self.text_projection(text_features).unsqueeze(1)
        image_proj = self.image_projection(image_features).unsqueeze(1)
        attn_output, _ = self.cross_attention(query=image_proj, key=text_proj, value=text_proj)
        fused = self.layer_norm(image_proj + attn_output)
        fused = self.final_fc(fused.squeeze(1))
        return fused, fused

# --- 2. 训练逻辑 (和之前一样，无需修改) ---

def contrastive_loss(text_embeds, image_embeds, temperature=0.07):
    logits = (text_embeds @ image_embeds.T) / temperature
    labels = torch.arange(len(text_embeds), device=text_embeds.device)
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

def train_model(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for text_batch, image_batch in progress_bar:
            text_batch, image_batch = text_batch.to(device), image_batch.to(device)
            optimizer.zero_grad()
            text_output, image_output = model(text_batch, image_batch)
            loss = contrastive_loss(text_output, image_output)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
    return model

def inference(model, dataloader, device):
    model.to(device)
    model.eval()
    all_fused_embeddings = []
    with torch.no_grad():
        for text_batch, image_batch in tqdm(dataloader, desc="Inference"):
            text_batch, image_batch = text_batch.to(device), image_batch.to(device)
            fused_embeds, _ = model(text_batch, image_batch)
            all_fused_embeddings.append(fused_embeds.cpu())
    return torch.cat(all_fused_embeddings, dim=0).numpy()


# --- 3. 主函数 (已修改) ---

def main():
    # --- 3.1 解析命令行参数 (MODIFIED) ---
    parser = argparse.ArgumentParser(description="融合文本和图像的 embedding 文件，支持多种方法。")
    parser.add_argument('--method', type=str, required=True, choices=['concat', 'clip-align', 'projection', 'cross-attention'], help='融合方法')
    
    # 新增的简化参数
    parser.add_argument('--base_path', type=str, default='../datasets', help='包含所有数据集文件夹的根目录')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (例如: Beauty, Sports)')
    parser.add_argument('--text_suffix', type=str, default='emb-qwen-td', help='文本特征的文件名后缀')
    parser.add_argument('--image_suffix', type=str, default='emb-ViT-L-14', help='图像特征的文件名后缀')
    
    # 训练和模型参数
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')
    parser.add_argument('--embed_dim', type=int, default=1024, help='共享空间的维度或最终输出维度')
    parser.add_argument('--nhead', type=int, default=4, help='交叉注意力模型的头数')
    args = parser.parse_args()

    # --- 3.2 自动构建路径 (NEW) ---
    dataset_folder = os.path.join(args.base_path, args.dataset_name)
    
    text_emb_path = os.path.join(dataset_folder, f"{args.dataset_name}.{args.text_suffix}.npy")
    image_emb_path = os.path.join(dataset_folder, f"{args.dataset_name}.{args.image_suffix}.npy")
    
    output_filename_base = f"{args.dataset_name}.emb-fused-{args.method}"
    output_path = os.path.join(dataset_folder, f"{output_filename_base}.npy")
    
    print("--- 自动构建路径 ---")
    print(f"文本特征路径: {text_emb_path}")
    print(f"图像特征路径: {image_emb_path}")
    print(f"融合输出路径: {output_path}")
    print("---------------------\n")

    # 检查输入文件是否存在
    if not os.path.exists(text_emb_path):
        print(f"错误: 找不到输入文件 {text_emb_path}")
        return
    if not os.path.exists(image_emb_path):
        print(f"错误: 找不到输入文件 {image_emb_path}")
        return

    # --- 3.3 执行选择的方法 ---
    if args.method == 'concat':
        print("方法: 直接拼接 (Concatenation)")
        text_embeddings = np.load(text_emb_path)
        image_embeddings = np.load(image_emb_path)
        
        if text_embeddings.shape[0] != image_embeddings.shape[0]:
            print("错误：物品数量不匹配！")
            return
            
        fused_embeddings = np.concatenate([text_embeddings, image_embeddings], axis=1)
        print(f"拼接完成. 融合后维度: {fused_embeddings.shape[1]}")
        
        np.save(output_path, fused_embeddings)
        print(f"已保存融合后的特征到: {output_path}")
        return

    # --- 需要训练的方法 ---
    print(f"方法: {args.method} (需要训练)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    text_features = torch.from_numpy(np.load(text_emb_path)).float()
    image_features = torch.from_numpy(np.load(image_emb_path)).float()
    
    dataset = TensorDataset(text_features, image_features)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    text_dim, image_dim = text_features.shape[1], image_features.shape[1]

    if args.method == 'clip-align':
        model = ClipAlignmentModel(text_dim, image_dim, args.embed_dim)
    elif args.method == 'projection':
        model = ProjectionFusionModel(text_dim, image_dim, args.embed_dim)
    elif args.method == 'cross-attention':
        model = CrossAttentionFusionModel(text_dim, image_dim, args.embed_dim, args.nhead)
    else:
        raise ValueError("未知的方法")

    print(f"\n--- 开始训练 {args.method} 模型 ---")
    trained_model = train_model(model, train_loader, args.epochs, args.lr, device)
    
    model_save_path = output_path.replace('.npy', '.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"\n训练好的模型已保存到: {model_save_path}")

    print("\n--- 开始使用训练好的模型进行推理 ---")
    inference_loader = DataLoader(dataset, batch_size=args.batch_size * 4, shuffle=False)
    final_embeddings = inference(trained_model, inference_loader, device)

    print(f"\n推理完成. 融合后特征 Shape: {final_embeddings.shape}")
    np.save(output_path, final_embeddings)
    print(f"已保存最终融合后的特征到: {output_path}")


if __name__ == '__main__':
    main()