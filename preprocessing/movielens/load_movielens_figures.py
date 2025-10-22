import os
import sys
import argparse
import requests
import json
from tqdm import tqdm
from collections import defaultdict
import time
import random

# 
def download_image(url, save_path, timeout=10):
    """下载图片并保存到指定路径"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {url}, 错误: {e}")
        return False

def is_valid_image(image_file):
    """检查图片文件是否有效"""
    if not os.path.exists(image_file):
        return False
    
    try:
        from PIL import Image
        with Image.open(image_file) as img:
            img.verify()
        return True
    except Exception:
        return False

def load_json(file):
    """加载JSON文件"""
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_movie_poster_url(movie_id, title, year=None):
    """
    这里集成TMDB API或者其他电影数据源
    """
    return None

def load_movielens_data(args):
    """加载MovieLens数据"""
    print('处理数据: ')
    print(' 数据集: ', args.dataset)
    
    # 构建路径
    processed_data_path = os.path.join(args.save_root, args.dataset)
    movies_file = os.path.join(processed_data_path, f'{args.dataset}.item.json')
    
    if not os.path.exists(movies_file):
        print(f"错误: 找不到电影数据文件 '{movies_file}'。请先运行数据预处理脚本。")
        sys.exit(1)
    
    movies_data = load_json(movies_file)
    print(f'找到 {len(movies_data)} 部电影。')
    
    return movies_data

def download_movie_posters(args, movies_data):
    """下载电影海报"""
    save_path = os.path.join(args.save_root, f'{args.dataset}/Images')
    item_images_file = os.path.join(args.save_root, f'{args.dataset}/Images', f'{args.dataset}_images_info.json')
    
    os.makedirs(save_path, exist_ok=True)
    
    if os.path.exists(item_images_file):
        item_images = load_json(item_images_file)
    else:
        item_images = defaultdict(list)
    
    download_count = 0
    processed_count = 0
    
    for movie_id, movie_info in tqdm(movies_data.items(), desc='处理电影海报'):
        if movie_id in item_images and len(item_images.get(movie_id, [])) > 0:
            continue
        
        title = movie_info.get('title', '')
        year = movie_info.get('year', None)
        
        # 尝试获取海报URL
        poster_url = get_movie_poster_url(movie_id, title, year)
        
        if poster_url:
            # 生成文件名
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            filename = f"{movie_id}_{safe_title}.jpg"
            save_file = os.path.join(save_path, filename)
            
            if not os.path.exists(save_file):
                if download_image(poster_url, save_file) and is_valid_image(save_file):
                    item_images[movie_id].append(filename)
                    download_count += 1
                elif os.path.exists(save_file):
                    os.remove(save_file)
            elif is_valid_image(save_file):
                item_images[movie_id].append(filename)
        else:
            # 没有海报URL，记录为空
            item_images[movie_id] = []
        
        processed_count += 1
        
        # 添加延迟避免请求过于频繁
        if processed_count % 10 == 0:
            time.sleep(random.uniform(0.5, 1.5))
    
    items_without_images = sum(1 for v in item_images.values() if not v)
    total_items = len(movies_data)
    
    print(f'\n本次新下载图片: {download_count}')
    print(f'总电影数: {total_items}')
    print(f'缺少图片的电影数: {items_without_images}')
    
    if total_items > 0:
        coverage = (total_items - items_without_images) / total_items
        print(f"图片覆盖率: {coverage:.2%}")
    else:
        print("图片覆盖率: 0.00% (没有找到任何有效电影)")
    
    # 保存图片信息
    with open(item_images_file, 'w', encoding='utf-8') as f:
        json.dump(item_images, f, indent=4, ensure_ascii=False)
    
    print(f"图片信息已保存到: {item_images_file}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="下载并处理MovieLens数据集的电影海报。")
    parser.add_argument('--dataset', type=str, default='ml-1m', help='数据集名称，例如 ml-1m, ml-10m, ml-20m')
    parser.add_argument('--save_root', type=str, default='../datasets', help='数据保存根目录')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("开始处理MovieLens电影海报...")
    print("注意: 由于MovieLens数据集本身不包含图片，此脚本主要用于演示框架。")
    print("实际应用中需要集成TMDB API来获取真实的电影海报。")
    
    movies_data = load_movielens_data(args)
    
    if movies_data:
        download_movie_posters(args, movies_data)
    else:
        print("\n没有找到任何需要处理的电影，程序结束。")
