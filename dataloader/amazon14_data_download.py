import os
import sys
import argparse
import requests
import gzip
import json
import csv
from tqdm import tqdm

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

def extract_ratings_from_reviews(reviews_gz_path: str, ratings_csv_path: str):
    """
    从 reviews.json.gz 文件中提取并保存 ratings 数据到 CSV 文件。
    """
    print(f"\n正在从 {os.path.basename(reviews_gz_path)} 提取 ratings 数据...")
    os.makedirs(os.path.dirname(ratings_csv_path), exist_ok=True)
    
    try:
        with gzip.open(reviews_gz_path, 'rt', encoding='utf-8') as f_in, \
             open(ratings_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            
            writer = csv.writer(f_out)
            
            for line in tqdm(f_in, desc="  -> 提取中"):
                try:
                    review = json.loads(line)
                    item_id = review.get('asin')
                    user_id = review.get('reviewerID')
                    rating = review.get('overall')
                    timestamp = review.get('unixReviewTime')
                    
                    if all([item_id, user_id, rating, timestamp]):
                        writer.writerow([item_id, user_id, rating, timestamp])
                except json.JSONDecodeError:
                    pass
        
        print(f"  -> 提取完成，ratings 数据已保存到: {ratings_csv_path}")
        return True
    except Exception as e:
        print(f"提取失败: {e}")
        if os.path.exists(ratings_csv_path):
            os.remove(ratings_csv_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="下载 Amazon 数据集并从 reviews 文件中提取 ratings。")
    parser.add_argument(
        '--category', 
        type=str, 
        required=True, 
        help="要处理的数据集类别，例如 'All_Beauty'、'Musical_Instruments' 或 'Baby'。"
    )
    parser.add_argument(
        '--data_version', 
        type=str, 
        default='14', 
        choices=['14', '18'], 
        help="数据集年份版本 (14 或 18)。"
    )
    parser.add_argument('--output_dir', type=str, default='../datasets', help="保存数据的根目录。")
    args = parser.parse_args()

    category = args.category
    data_version = args.data_version
    
    base_output_dir = os.path.join(args.output_dir, f'amazon{data_version}')
    base_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
    
    metadata_dir = os.path.join(base_output_dir, 'Metadata')
    ratings_dir = os.path.join(base_output_dir, 'Ratings')
    review_dir = os.path.join(base_output_dir, 'Review')

    meta_gz_filename = f'meta_{category}.json.gz'
    # 服务端文件名，14年数据带有 'reviews_' 前缀，18年数据没有
    review_gz_server_filename = f'reviews_{category}_5.json.gz' if data_version == '14' else f'{category}_5.json.gz'
    # 本地统一保存的文件名（统一为18年格式）
    review_gz_local_filename = f'{category}_5.json.gz'
    ratings_csv_filename = f'{category}.csv'

    print("="*50)
    print(f"开始处理类别: {category}")
    print(f"数据集版本: {data_version}")
    print(f"数据将保存在: {os.path.abspath(base_output_dir)}")
    print("="*50)

    # --- 1. 下载元数据文件 (Metadata) ---
    meta_gz_path = os.path.join(metadata_dir, meta_gz_filename)
    meta_url = f'{base_url}{meta_gz_filename}'
    if not os.path.exists(meta_gz_path):
        if not download_file(meta_url, meta_gz_path):
            sys.exit(1)
    else:
        print(f"文件已存在，跳过下载: {meta_gz_path}")

    # --- 2. 下载并重命名评论文件 (Review) ---
    review_gz_temp_path = os.path.join(review_dir, review_gz_server_filename)
    review_gz_final_path = os.path.join(review_dir, review_gz_local_filename)
    review_url = f'{base_url}{review_gz_server_filename}'
    
    if not os.path.exists(review_gz_final_path):
        print("\n开始处理评论文件...")
        # 步骤 2.1: 下载文件（使用服务器上的原始文件名）
        if not os.path.exists(review_gz_temp_path):
            if not download_file(review_url, review_gz_temp_path):
                sys.exit(1)
        else:
            print(f"文件已存在，跳过下载: {review_gz_temp_path}")
        
        # 步骤 2.2: 重命名文件
        try:
            os.rename(review_gz_temp_path, review_gz_final_path)
            print(f"  -> 文件已重命名为: {review_gz_local_filename}")
        except FileNotFoundError:
            print(f"重命名失败: 找不到文件 {review_gz_temp_path}")
            sys.exit(1)
            
    else:
        print(f"\n文件已存在，跳过下载和重命名: {review_gz_final_path}")

    # --- 3. 从下载的评论文件中提取 ratings ---
    ratings_csv_path = os.path.join(ratings_dir, ratings_csv_filename)
    if not os.path.exists(ratings_csv_path):
        if not extract_ratings_from_reviews(review_gz_final_path, ratings_csv_path):
            sys.exit(1)
    else:
        print(f"\nratings 文件已存在，跳过提取: {ratings_csv_path}")
        
    print("\n所有任务完成！")
    print("最终生成的文件结构如下:")
    print(f"  - {os.path.abspath(metadata_dir)}/{meta_gz_filename}")
    print(f"  - {os.path.abspath(review_dir)}/{review_gz_local_filename}")
    print(f"  - {os.path.abspath(ratings_dir)}/{ratings_csv_filename}")

if __name__ == '__main__':
    main()