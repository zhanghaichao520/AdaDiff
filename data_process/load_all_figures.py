import os
import sys
import argparse
import requests
import gzip
import json
import ast  # <-- 引入 ast 模块
from tqdm import tqdm
from collections import defaultdict

# --- Helper Functions (无需修改) ---
def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    except requests.exceptions.RequestException:
        return False

def is_valid_jpg(jpg_file):
    if not os.path.exists(jpg_file): return False
    with open(jpg_file, 'rb') as f:
        file_size = os.path.getsize(jpg_file)
        if file_size < 2: return False
        f.seek(file_size - 2)
        return f.read() == b'\xff\xd9'

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

# --- Core Logic Functions (已修改) ---

def load_meta_items(file, data_version):
    """
    加载元数据文件，智能兼容 2014 和 2018 年的数据集格式。
    """
    items = {}
    # 2014年的文件需要用 ast.literal_eval 解析，2018年的用 json.loads
    # 我们根据版本选择不同的解析函数
    parser = ast.literal_eval if data_version == '14' else json.loads

    with gzip.open(file, 'rt', encoding='utf-8') as fp:
        for line in tqdm(fp, desc="Load metas"):
            try:
                # 使用基于版本的解析器
                data = parser(line)
                item_id = data.get("asin")
                
                if not item_id:
                    continue

                unified_info = {'imageURLHighRes': []}

                # 兼容 2018 年的数据格式
                if 'imageURLHighRes' in data:
                    urls = data.get('imageURLHighRes', [])
                    if isinstance(urls, list):
                        unified_info['imageURLHighRes'] = urls
                # 兼容 2014 年的数据格式
                elif 'imUrl' in data:
                    im_url = data.get('imUrl')
                    if im_url and isinstance(im_url, str):
                        unified_info['imageURLHighRes'] = [im_url]
                
                items[item_id] = unified_info
            
            except (ValueError, SyntaxError, TypeError):
                # 捕获 ast.literal_eval 和 json.loads 可能的错误
                continue
    return items

def load_meta_data(args):
    """加载元数据文件并返回物品信息。"""
    print('Process data: ')
    print(' Dataset: ', args.dataset)
    print(' Data Version: ', args.data_version)
    
    meta_file_path = os.path.join(
        args.base_path, f'amazon{args.data_version}/Metadata', f'meta_{args.dataset}.json.gz'
    )
    
    if not os.path.exists(meta_file_path):
        print(f"错误: 找不到元数据文件 '{meta_file_path}'。请检查路径和文件名。")
        sys.exit(1)
        
    # 将 data_version 传递给解析函数
    meta_items = load_meta_items(meta_file_path, args.data_version)
    return meta_items

def load_5_core_review_items(args, all_meta_items):
    """加载 review 文件并过滤出在 review 数据中出现过的物品。"""
    if not all_meta_items: # 如果没有元数据，直接返回空字典
        return {}

    review_file_path = os.path.join(
        args.base_path, f'amazon{args.data_version}/Review', f'{args.dataset}_5.json.gz'
    )
    
    if not os.path.exists(review_file_path):
        print(f"错误: 找不到 review 文件 '{review_file_path}'。")
        sys.exit(1)

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

def download_and_process_images(args, items_to_process):
    """主函数，用于下载和处理图像。"""
    save_path = os.path.join(args.base_path, f'amazon{args.data_version}/Images', args.dataset)
    item_images_file = os.path.join(args.base_path, f'amazon{args.data_version}/Images', f'{args.dataset}_images_info.json')
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(item_images_file):
        item_images = load_json(item_images_file)
    else:
        item_images = defaultdict(list)
    
    download_count = 0
    
    for asin, info in tqdm(items_to_process.items(), desc='Downloading images'):
        if asin in item_images and len(item_images.get(asin, [])) > 0:
            continue
        
        image_urls = info.get('imageURLHighRes', [])
        
        image_downloaded = False
        for image_url in image_urls:
            if not isinstance(image_url, str) or not image_url.startswith('http'):
                continue

            name = os.path.basename(image_url)
            save_file = os.path.join(save_path, name)
            
            if not os.path.exists(save_file):
                if download_image(image_url, save_file) and is_valid_jpg(save_file):
                    item_images[asin].append(name)
                    image_downloaded = True
                    download_count += 1
                    break 
                elif os.path.exists(save_file):
                    os.remove(save_file)
            elif is_valid_jpg(save_file):
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
    
    # --- 新增：防止除以零的保护 ---
    if total_items > 0:
        coverage = (total_items - items_without_images) / total_items
        print(f"图片覆盖率: {coverage:.2%}")
    else:
        print("图片覆盖率: 0.00% (没有找到任何有效物品)")
    
    with open(item_images_file, 'w', encoding='utf8') as item_images_f:
        json.dump(item_images, item_images_f, indent=4)

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="下载并验证Amazon数据集的商品图片。")
    parser.add_argument('--dataset', type=str, default='Baby', help='例如 Baby, All_Beauty, Musical_Instruments 等')
    parser.add_argument('--data_version', type=str, default='14', choices=['14', '18'], help='数据集年份版本 (14 或 18)')
    parser.add_argument('--base_path', type=str, default='../datasets')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    all_meta_items = load_meta_data(args)
    print(f'Meta 文件中总共找到 {len(all_meta_items)} 个物品。')
    
    filtered_items = load_5_core_review_items(args, all_meta_items)
    print(f'过滤后，保留 {len(filtered_items)} 个在 review 中出现过的物品进行图片下载。')

    if filtered_items:
        download_and_process_images(args, filtered_items)
    else:
        print("\n没有找到任何需要处理的物品，程序结束。")