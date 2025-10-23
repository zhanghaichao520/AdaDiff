import os
import sys
import argparse
import requests
import gzip
import json
import ast  # 用于 Amazon 2014
import time
import random
from tqdm import tqdm
from collections import defaultdict
from PIL import Image # 用于 is_valid_image

# =================================================================
# ================== 共享的工具函数 ==================
# =================================================================

def download_image(url, save_path, timeout=10):
    """(共享) 下载图片并保存到指定路径"""
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
        # print(f"下载失败: {url}, 错误: {e}") # 打印过多，暂时注释
        return False

def is_valid_image(image_file):
    """(共享) 检查图片文件是否有效 (使用 PIL)"""
    if not os.path.exists(image_file):
        return False
    
    # 检查文件大小是否为0
    if os.path.getsize(image_file) == 0:
        return False
        
    try:
        with Image.open(image_file) as img:
            img.verify() # 验证图片头部
        return True
    except Exception: # (IOError, SyntaxError, etc.)
        return False

def load_json(file):
    """(共享) 加载JSON文件"""
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# =================================================================
# ================== Amazon 专属逻辑 ==================
# =================================================================

def load_meta_items_amazon(file, data_version):
    """(Amazon) 加载元数据文件，智能兼容 2014 和 2018。"""
    items = {}
    parser = ast.literal_eval if data_version == '14' else json.loads

    with gzip.open(file, 'rt', encoding='utf-8') as fp:
        for line in tqdm(fp, desc="Load Amazon metas"):
            try:
                data = parser(line)
                item_id = data.get("asin")
                if not item_id:
                    continue

                unified_info = {'imageURLHighRes': []}

                # 兼容 2018
                if 'imageURLHighRes' in data:
                    urls = data.get('imageURLHighRes', [])
                    if isinstance(urls, list):
                        unified_info['imageURLHighRes'] = urls
                # 兼容 2014
                elif 'imUrl' in data:
                    im_url = data.get('imUrl')
                    if im_url and isinstance(im_url, str):
                        unified_info['imageURLHighRes'] = [im_url]
                
                items[item_id] = unified_info
            
            except (ValueError, SyntaxError, TypeError):
                continue
    return items

def load_5_core_review_items_amazon(args, all_meta_items):
    """(Amazon) 加载 review 文件并过滤出在 review 数据中出现过的物品。"""
    if not all_meta_items:
        return {}

    review_file_path = os.path.join(
        args.data_root, f'amazon{args.data_version}/Review', f'{args.dataset}_5.json.gz'
    )
    
    if not os.path.exists(review_file_path):
        print(f"警告: 找不到 review 文件 '{review_file_path}'。将尝试下载所有元数据中的图片。")
        # 如果找不到 5-core review 文件，就退而求其次，下载所有 meta 里的
        return all_meta_items

    review_items = {}
    with gzip.open(review_file_path, 'rt', encoding='utf-8') as gin:
        for line in tqdm(gin, desc='Filtering items with reviews'):
            try:
                review_data = json.loads(line)
                item_id = review_data.get('asin')
                if item_id and item_id in all_meta_items and item_id not in review_items:
                    review_items[item_id] = all_meta_items[item_id]
            except (json.JSONDecodeError, KeyError):
                continue
                
    return review_items

def download_and_process_images_amazon(args, items_to_process):
    """(Amazon) 主函数，用于下载和处理图像。"""
    save_path = os.path.join(args.data_root, f'amazon{args.data_version}/Images', args.dataset)
    item_images_file = os.path.join(args.data_root, f'amazon{args.data_version}/Images', f'{args.dataset}_images_info.json')
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(item_images_file):
        item_images = load_json(item_images_file)
    else:
        item_images = defaultdict(list)
    
    download_count = 0
    
    for asin, info in tqdm(items_to_process.items(), desc='Downloading Amazon images'):
        if asin in item_images and len(item_images.get(asin, [])) > 0:
            continue
        
        image_urls = info.get('imageURLHighRes', [])
        image_downloaded = False
        
        for image_url in image_urls:
            if not isinstance(image_url, str) or not image_url.startswith('http'):
                continue

            name = os.path.basename(image_url)
            # 避免文件名过长或包含非法字符 (虽然 Amazon URL 通常还好)
            if len(name) > 200:
                 name = name[-200:]
            
            save_file = os.path.join(save_path, name)
            
            if not os.path.exists(save_file):
                # 使用共享的 download_image 和 is_valid_image
                if download_image(image_url, save_file) and is_valid_image(save_file):
                    item_images[asin].append(name)
                    image_downloaded = True
                    download_count += 1
                    break 
                elif os.path.exists(save_file):
                    # 下载失败或文件无效，删除
                    os.remove(save_file)
            elif is_valid_image(save_file):
                item_images[asin].append(name)
                image_downloaded = True
                break
        
        if not image_downloaded:
            item_images[asin] = []
            
    items_without_images = sum(1 for v in item_images.values() if not v)
    total_items = len(items_to_process)

    print(f'\n本次新下载图片: {download_count}')
    print(f'总物品数: {total_items}')
    print(f'缺少图片/URL无效的物品数: {items_without_images}')
    
    if total_items > 0:
        coverage = (total_items - items_without_images) / total_items
        print(f"图片覆盖率: {coverage:.2%}")
    else:
        print("图片覆盖率: 0.00% (没有找到任何有效物品)")
    
    with open(item_images_file, 'w', encoding='utf8') as item_images_f:
        json.dump(item_images, item_images_f, indent=4)

def run_amazon_download(args):
    """(Amazon) 运行亚马逊图片下载的主流程"""
    print(f'处理 Amazon 数据集: {args.dataset} (Version: {args.data_version})')
    
    meta_file_path = os.path.join(
        args.data_root, f'amazon{args.data_version}/Metadata', f'meta_{args.dataset}.json.gz'
    )
    if not os.path.exists(meta_file_path):
        print(f"错误: 找不到元数据文件 '{meta_file_path}'。请检查路径和文件名。")
        sys.exit(1)
        
    all_meta_items = load_meta_items_amazon(meta_file_path, args.data_version)
    print(f'Meta 文件中总共找到 {len(all_meta_items)} 个物品。')
    
    filtered_items = load_5_core_review_items_amazon(args, all_meta_items)
    print(f'过滤后，保留 {len(filtered_items)} 个在 review 中出现过的物品进行图片下载。')

    if filtered_items:
        download_and_process_images_amazon(args, filtered_items)
    else:
        print("\n没有找到任何需要处理的物品，程序结束。")

# =================================================================
# ================== MovieLens 专属逻辑 ==================
# =================================================================

def get_movie_poster_url_movielens(movie_id, title, year=None):
    """
    (MovieLens) 获取电影海报URL的占位符函数。
    
    !!! 警告: 此函数需要你自己实现 !!!
    
    你需要:
    1. 注册一个电影数据库API (例如 TMDB: https://www.themoviedb.org/documentation/api)
    2. 获取你的 API 密钥 (API Key)。
    3. 在这里编写代码，使用 title 和 year 去 API 搜索电影。
    4. 从 API 响应中提取海报图片的 URL (poster_path)。
    5. 返回完整的 URL (例如 "https://image.tmdb.org/t/p/w500/[poster_path]")。
    
    为了演示，这里将始终返回 None。
    """
    # print("警告: get_movie_poster_url_movielens() 未实现。无法下载 MovieLens 海报。")
    # raise NotImplementedError("你需要实现 get_movie_poster_url_movielens 函数来从外部 API 获取海报 URL。")
    return None # 返回 None 以允许脚本继续运行（但不下载）

def load_movielens_data_movielens(args):
    """(MovieLens) 加载 MovieLens 数据"""
    print('处理 MovieLens 数据: ')
    print(' 数据集: ', args.dataset)
    
    processed_data_path = os.path.join(args.data_root, args.dataset)
    movies_file = os.path.join(processed_data_path, f'{args.dataset}.item.json')
    
    if not os.path.exists(movies_file):
        print(f"错误: 找不到电影数据文件 '{movies_file}'。请先运行数据预处理脚本。")
        sys.exit(1)
    
    movies_data = load_json(movies_file)
    print(f'找到 {len(movies_data)} 部电影 (来自 .item.json)。')
    
    return movies_data

def download_movie_posters_movielens(args, movies_data):
    """(MovieLens) 下载电影海报"""
    save_path = os.path.join(args.data_root, f'{args.dataset}/Images')
    item_images_file = os.path.join(args.data_root, f'{args.dataset}/Images', f'{args.dataset}_images_info.json')
    
    os.makedirs(save_path, exist_ok=True)
    
    if os.path.exists(item_images_file):
        item_images = load_json(item_images_file)
    else:
        item_images = defaultdict(list)
    
    download_count = 0
    processed_count = 0
    
    # 检查一次占位符函数
    if get_movie_poster_url_movielens("test_id", "test_title", "2000") is None:
        print("\n" + "="*30)
        print("警告: 'get_movie_poster_url_movielens' 函数未实现或返回 None。")
        print("脚本将继续运行以生成空的 images_info.json 文件，但不会下载任何图片。")
        print("请编辑此脚本以集成 TMDB 或其他 API。")
        print("="*30 + "\n")
        time.sleep(3) # 给用户时间阅读警告
    
    for movie_id, movie_info in tqdm(movies_data.items(), desc='处理 MovieLens 海报'):
        # movie_id 在 .item.json 中是 0-based index (字符串形式)
        # 我们需要原始ID（如果存储在 movie_info 中）或标题/年份来进行搜索
        
        if movie_id in item_images and len(item_images.get(movie_id, [])) > 0:
            continue
        
        title = movie_info.get('title', '')
        year = movie_info.get('year', None)
        
        # 尝试获取海报URL
        poster_url = get_movie_poster_url_movielens(movie_id, title, year)
        
        if poster_url:
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            # 使用 movie_id (0-based index) 作为文件名更安全
            filename = f"{movie_id}.jpg"
            save_file = os.path.join(save_path, filename)
            
            if not os.path.exists(save_file):
                # 使用共享的 download_image 和 is_valid_image
                if download_image(poster_url, save_file) and is_valid_image(save_file):
                    item_images[movie_id].append(filename)
                    download_count += 1
                elif os.path.exists(save_file):
                    os.remove(save_file)
            elif is_valid_image(save_file):
                item_images[movie_id].append(filename)
        else:
            # 没有海报URL
            item_images[movie_id] = []
        
        processed_count += 1
        
        # 添加延迟避免请求过于频繁 (如果实现了API)
        if poster_url and processed_count % 10 == 0:
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

def run_movielens_download(args):
    """(MovieLens) 运行 MovieLens 图片下载的主流程"""
    movies_data = load_movielens_data_movielens(args)
    
    if movies_data:
        download_movie_posters_movielens(args, movies_data)
    else:
        print("\n没有找到任何需要处理的电影，程序结束。")

# =================================================================
# =================== 主程序入口和参数解析 ===================
# =================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="下载并处理 Amazon 或 MovieLens 数据集的图片。")
    
    # --- 必需参数 ---
    parser.add_argument('--dataset_type', type=str, required=True, choices=['amazon', 'movielens'],
                        help='要处理的数据集类型 (amazon 或 movielens)')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='数据集名称 (例如: Home, Baby, ml-1m, ml-20m)')
    
    # --- 通用路径参数 ---
    parser.add_argument('--data_root', type=str, default='../datasets', 
                        help='数据保存的根目录 (包含 amazon14/, ml-1m/ 等文件夹)')
    
    # --- Amazon 专用参数 ---
    parser.add_argument('--data_version', type=str, default='14', choices=['14', '18'],
                        help='Amazon 数据集年份版本 (14 或 18)。仅当 dataset_type=amazon 时使用。')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print('\n' + '=' * 20)
    print(f"开始下载图片: {args.dataset} (Type: {args.dataset_type})")
    print('=' * 20 + '\n')
    
    # 根据 dataset_type 调度不同的任务
    if args.dataset_type == 'amazon':
        run_amazon_download(args)
    elif args.dataset_type == 'movielens':
        run_movielens_download(args)
    
    print(f"\n图片处理完成: {args.dataset}")