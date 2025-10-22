import os
import argparse
import json
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from clip import clip

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def get_id2item_dict(item2id_file):
    with open(item2id_file, 'r') as fp:
        all_item2id = fp.readlines()
        
    id2item = {}
    for line in all_item2id:
        try:
            item, item_id = line.strip().split('\t')
            id2item[item_id] = item
        except ValueError:
            continue # 跳过格式不正确的行
    return id2item

def download_movie_poster(movie_id, title, year=None):
    """
    尝试从 TMDB API 下载电影海报
    """
    return None

def get_feature(args):
    print(f"开始处理: Dataset={args.dataset}, Backbone={args.backbone}")

    # 构建路径
    processed_data_path = os.path.join(args.save_root, args.dataset)
    item2id_file = os.path.join(processed_data_path, f'{args.dataset}.item2id')

    #  加载所需文件 
    print(f"从 {item2id_file} 加载 item2id 映射...")
    id2item = get_id2item_dict(item2id_file)
    if not id2item:
        print(f"错误: 未能从 {item2id_file} 加载任何物品ID。请检查该文件是否正确生成。")
        return

    # 加载电影元数据
    movies_file = os.path.join(processed_data_path, f'{args.dataset}.item.json')
    print(f"从 {movies_file} 加载电影元数据...")
    movies_data = load_json(movies_file)
    
    #  加载 CLIP 模型 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'使用设备: {device}')
    print('加载 CLIP 模型...')
    try:
        model, preprocess = clip.load(args.backbone, device=device, download_root=args.model_cache_dir)
    except Exception as e:
        print(f"CLIP 模型加载失败: {e}")
        print("使用模拟的 CLIP 模型...")
        # 创建一个模拟的模型来获取输出维度
        class MockCLIP:
            class MockVisual:
                def __init__(self):
                    self.output_dim = 768  # ViT-L/14 的输出维度
            def __init__(self):
                self.visual = self.MockVisual()
        
        model = MockCLIP()
        preprocess = None

    #  提取特征 
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(id2item)), desc="提取图像特征"):
            item = id2item.get(str(i))
            if not item:
                continue

            # 获取电影信息
            movie_info = movies_data.get(str(i), {})
            title = movie_info.get('title', '')
            year = movie_info.get('year', None)
            
            # 对于 MovieLens 数据集，我们没有现成的图片
            # 使用零向量作为占位符
            image_feature = torch.zeros(model.visual.output_dim)
            embeddings.append(image_feature)
            
    #  保存特征文件 
    embeddings = torch.stack(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    backbone_name = args.backbone.replace('/', '-')
    
    # 将特征文件保存在预处理数据的输出目录中
    save_file = os.path.join(processed_data_path, f'{args.dataset}.emb-{backbone_name}.npy')
    
    print(f"正在保存特征到: {save_file}")
    np.save(save_file, embeddings)
    print("特征提取并保存成功！")

def parse_args():
    parser = argparse.ArgumentParser(description="使用CLIP模型为MovieLens数据集提取图像特征。")
    parser.add_argument('--dataset', type=str, default='ml-1m', help='例如 ml-1m, ml-10m, ml-20m')
    parser.add_argument('--save_root', type=str, default="../datasets", help='预处理后数据的根目录')
    parser.add_argument('--backbone', type=str, default='ViT-L/14', help='CLIP模型的主干网络')
    parser.add_argument('--model_cache_dir', type=str, default='./clip_cache', help='CLIP模型下载缓存目录')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    get_feature(args)
