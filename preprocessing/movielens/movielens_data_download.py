import os
import sys
import argparse
import requests
import zipfile
import csv
from tqdm import tqdm
import json

def download_file(url: str, filepath: str):
    """
    使用 requests 和 tqdm 下载文件并显示进度条。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"准备下载: {os.path.basename(filepath)}")
    print(f"从: {url}")
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="  -> 下载中")
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            print(f"  -> 下载完成，已保存到: {filepath}")
            return True
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def extract_zip_file(zip_path: str, extract_to: str):
    """
    解压 zip 文件到指定目录。
    """
    print(f"正在解压: {os.path.basename(zip_path)}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  -> 解压完成，已保存到: {extract_to}")
        return True
    except zipfile.BadZipFile as e:
        print(f"解压失败: {e}")
        return False

def process_movielens_data(dataset_dir: str, output_dir: str, dataset_name: str):
    """
    处理 MovieLens 数据，生成评分文件和电影元数据文件。
    """
    print(f"处理 MovieLens 数据: {dataset_name}")
    
    # 查找 ratings.dat 和 movies.dat 文件
    ratings_file = None
    movies_file = None
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file == 'ratings.dat':
                ratings_file = os.path.join(root, file)
            elif file == 'movies.dat':
                movies_file = os.path.join(root, file)
    
    if not ratings_file:
        print("错误: 找不到 ratings.dat 文件")
        return False
    
    if not movies_file:
        print("错误: 找不到 movies.dat 文件")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理评分数据
    print("处理评分数据...")
    ratings_output = os.path.join(output_dir, f'{dataset_name}.csv')
    
    with open(ratings_file, 'r', encoding='utf-8') as f_in, \
         open(ratings_output, 'w', newline='', encoding='utf-8') as f_out:
        
        writer = csv.writer(f_out)
        
        for line in tqdm(f_in, desc="处理评分数据"):
            line = line.strip()
            if line:
                # MovieLens 使用 :: 作为分隔符
                parts = line.split('::')
                if len(parts) >= 4:
                    user_id, movie_id, rating, timestamp = parts[0], parts[1], parts[2], parts[3]
                    # 重新排列为: movie_id, user_id, rating, timestamp
                    writer.writerow([movie_id, user_id, rating, timestamp])
    
    print(f"评分数据已保存到: {ratings_output}")
    
    # 处理电影元数据
    print("处理电影元数据...")
    movies_output = os.path.join(output_dir, f'{dataset_name}_movies.json')
    movies_data = {}
    
    with open(movies_file, 'r', encoding='latin-1') as f:
        for line in tqdm(f, desc="处理电影元数据"):
            line = line.strip()
            if line:
                # MovieLens 使用 :: 作为分隔符
                parts = line.split('::')
                if len(parts) >= 3:
                    movie_id, title, genres = parts[0], parts[1], parts[2]
                    
                    # 解析年份（从标题中提取）
                    year = None
                    title_clean = title
                    if '(' in title and ')' in title:
                        year_part = title[title.rfind('(')+1:title.rfind(')')]
                        if year_part.isdigit() and len(year_part) == 4:
                            year = int(year_part)
                            title_clean = title[:title.rfind('(')].strip()
                    
                    movies_data[movie_id] = {
                        'title': title_clean,
                        'year': year,
                        'genres': genres.split('|') if genres else [],
                        'description': f"{title_clean} ({year})" if year else title_clean
                    }
    
    with open(movies_output, 'w', encoding='utf-8') as f:
        json.dump(movies_data, f, indent=2, ensure_ascii=False)
    
    print(f"电影元数据已保存到: {movies_output}")
    return True

def main():
    parser = argparse.ArgumentParser(description="下载 MovieLens 数据集并处理数据。")
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        choices=['ml-1m', 'ml-10m', 'ml-20m'],
        help="要下载的 MovieLens 数据集版本"
    )
    parser.add_argument('--output_dir', type=str, default='../datasets', help="保存数据的根目录。")
    args = parser.parse_args()

    dataset = args.dataset
    
    # MovieLens 数据集 URL 映射
    dataset_urls = {
        'ml-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'ml-10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
        'ml-20m': 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'
    }
    
    base_output_dir = os.path.join(args.output_dir, dataset)
    temp_dir = os.path.join(base_output_dir, 'temp')
    final_dir = os.path.join(base_output_dir, 'processed')
    
    zip_filename = f'{dataset}.zip'
    zip_path = os.path.join(temp_dir, zip_filename)
    url = dataset_urls[dataset]

    print("="*50)
    print(f"开始处理数据集: {dataset}")
    print(f"数据将保存在: {os.path.abspath(base_output_dir)}")
    print("="*50)

    # --- 1. 下载数据集 ---
    if not os.path.exists(zip_path):
        if not download_file(url, zip_path):
            sys.exit(1)
    else:
        print(f"文件已存在，跳过下载: {zip_path}")

    # --- 2. 解压数据集 ---
    if not os.path.exists(os.path.join(temp_dir, dataset)):
        if not extract_zip_file(zip_path, temp_dir):
            sys.exit(1)
    else:
        print(f"文件已存在，跳过解压")

    # --- 3. 处理数据 ---
    dataset_dir = os.path.join(temp_dir, dataset)
    if not process_movielens_data(dataset_dir, final_dir, dataset):
        sys.exit(1)
        
    print("\n所有任务完成！")
    print("最终生成的文件结构如下:")
    print(f"  - {os.path.abspath(final_dir)}/{dataset}.csv")
    print(f"  - {os.path.abspath(final_dir)}/{dataset}_movies.json")

if __name__ == '__main__':
    main()
