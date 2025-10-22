import os
import sys
import argparse
import requests
import gzip
import json
import csv
import zipfile
from tqdm import tqdm

# --- 通用輔助函數 ---

def download_file(url: str, filepath: str, description: str = "Downloading"):
    """
    使用 requests 和 tqdm 下载文件并显示进度条。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"准备下载: {os.path.basename(filepath)}")
    print(f"从: {url}")

    if os.path.exists(filepath):
        print(f"文件已存在，跳过下载: {filepath}")
        return True

    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 8192
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"  -> {description}")
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                 print("警告: 下载的文件大小与 Content-Length 不符。")
            print(f"  -> 下载完成，已保存到: {filepath}")
            return True
    except requests.exceptions.Timeout:
        print(f"下载超时: {url}")
        if os.path.exists(filepath): os.remove(filepath)
        return False
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        if os.path.exists(filepath): os.remove(filepath)
        return False

def extract_zip_file(zip_path: str, extract_to: str):
    """
    解压 zip 文件到指定目录。
    """
    print(f"正在解压: {os.path.basename(zip_path)}")
    os.makedirs(extract_to, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 檢查 zip 文件內容，避免 Zip Slip 攻擊 (雖然官方源不太可能)
            for member in zip_ref.namelist():
                 member_path = os.path.join(extract_to, member)
                 abs_member_path = os.path.abspath(member_path)
                 abs_extract_to = os.path.abspath(extract_to)
                 if not abs_member_path.startswith(abs_extract_to):
                     raise SecurityException(f"非法成员路径: {member}")
            # 安全解压
            zip_ref.extractall(extract_to)
        print(f"  -> 解压完成，已保存到: {extract_to}")
        return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        print(f"解压失败: {e}")
        return False
    except NameError: # SecurityException is not a built-in type
         print(f"解压失败: 检测到非法成员路径，可能存在安全风险。")
         return False
    except Exception as e: # 捕获其他可能的解压错误
         print(f"解压时发生未知错误: {e}")
         return False


def extract_ratings_from_amazon_reviews(reviews_gz_path: str, ratings_csv_path: str):
    """
    (Amazon 特定) 从 reviews.json.gz 提取 ratings 到 CSV。
    """
    print(f"\n正在从 {os.path.basename(reviews_gz_path)} 提取 Amazon ratings 数据...")
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
                    if all([item_id, user_id, rating is not None, timestamp is not None]):
                        writer.writerow([item_id, user_id, rating, timestamp]) # asin, reviewerID, overall, unixReviewTime
                except (json.JSONDecodeError, TypeError):
                    pass # 跳过格式错误的行
        print(f"  -> 提取完成，ratings 数据已保存到: {ratings_csv_path}")
        return True
    except FileNotFoundError:
        print(f"错误: 找不到 Review 文件: {reviews_gz_path}")
        return False
    except Exception as e:
        print(f"提取失败: {e}")
        if os.path.exists(ratings_csv_path): os.remove(ratings_csv_path)
        return False

# --- 特定数据源处理逻辑 ---

def process_amazon(dataset_name: str, data_version: str, output_dir: str):
    """
    处理 Amazon 数据集的下载和初步提取。
    """
    print("\n" + "="*15 + f" 处理 Amazon 数据集: {dataset_name} (v{data_version}) " + "="*15)

    category = dataset_name # Amazon 使用 category 名称
    base_output_dir = os.path.join(output_dir, f'amazon{data_version}')
    base_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'

    metadata_dir = os.path.join(base_output_dir, 'Metadata')
    ratings_dir = os.path.join(base_output_dir, 'Ratings')
    review_dir = os.path.join(base_output_dir, 'Review')

    meta_gz_filename = f'meta_{category}.json.gz'
    review_gz_server_filename = f'reviews_{category}_5.json.gz' if data_version == '14' else f'{category}_5.json.gz'
    review_gz_local_filename = f'{category}_5.json.gz' # 统一本地名
    ratings_csv_filename = f'{category}.csv'

    print(f"数据将保存在: {os.path.abspath(base_output_dir)}")

    # --- 1. 下载元数据 ---
    meta_gz_path = os.path.join(metadata_dir, meta_gz_filename)
    meta_url = f'{base_url}{meta_gz_filename}'
    if not download_file(meta_url, meta_gz_path, description="下载元数据"):
        return False # 下载失败则停止

    # --- 2. 下载并重命名评论文件 ---
    review_gz_temp_path = os.path.join(review_dir, review_gz_server_filename)
    review_gz_final_path = os.path.join(review_dir, review_gz_local_filename)
    review_url = f'{base_url}{review_gz_server_filename}'

    if not os.path.exists(review_gz_final_path):
        print("\n开始处理评论文件...")
        if not os.path.exists(review_gz_temp_path):
             if not download_file(review_url, review_gz_temp_path, description="下载评论"):
                  return False
        else:
             print(f"临时文件已存在: {review_gz_temp_path}")

        try:
            os.rename(review_gz_temp_path, review_gz_final_path)
            print(f"  -> 文件已重命名为: {review_gz_local_filename}")
        except OSError as e:
            print(f"重命名失败: {e}")
            # 如果重命名失败，但最终文件已存在（可能上次运行中断），也算成功
            if not os.path.exists(review_gz_final_path):
                 return False
    else:
        print(f"\n评论文件已存在，跳过下载和重命名: {review_gz_final_path}")

    # --- 3. 提取 Ratings ---
    ratings_csv_path = os.path.join(ratings_dir, ratings_csv_filename)
    if not os.path.exists(ratings_csv_path):
        if not extract_ratings_from_amazon_reviews(review_gz_final_path, ratings_csv_path):
            return False
    else:
        print(f"\nRatings 文件已存在，跳过提取: {ratings_csv_path}")

    print("\nAmazon 数据下载和初步提取完成！")
    print("生成的文件:")
    print(f"  - {os.path.abspath(meta_gz_path)}")
    print(f"  - {os.path.abspath(review_gz_final_path)}")
    print(f"  - {os.path.abspath(ratings_csv_path)}")
    return True


def process_movielens(dataset_name: str, output_dir: str):
    """
    处理 MovieLens 数据集的下载、解压和格式转换。
    """
    print("\n" + "="*15 + f" 处理 MovieLens 数据集: {dataset_name} " + "="*15)

    # MovieLens 数据集 URL 映射
    dataset_urls = {
        'ml-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'ml-10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip', # 注意 ml-10m 文件名可能不同
        'ml-20m': 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'
    }

    if dataset_name not in dataset_urls:
        print(f"错误: 不支持的 MovieLens 数据集 '{dataset_name}'")
        return False

    url = dataset_urls[dataset_name]
    # 使用 dataset_name 作为主目录名
    base_output_dir = os.path.join(output_dir, dataset_name)
    temp_dir = os.path.join(base_output_dir, 'raw') # 将原始解压文件放入 raw 子目录
    final_dir = os.path.join(base_output_dir, 'processed') # 最终处理好的文件放入 processed 子目录

    zip_filename = f'{dataset_name}.zip'
    zip_path = os.path.join(temp_dir, zip_filename) # zip 文件也放入 raw
    # 解压后的目录名通常与 zip 文件名类似，但不带 .zip
    extracted_folder_name_pattern = dataset_name.replace('.zip', '') # 简单处理

    print(f"数据将保存在: {os.path.abspath(base_output_dir)}")

    # --- 1. 下载 ---
    if not download_file(url, zip_path, description=f"下载 {dataset_name}"):
        return False

    # --- 2. 解压 ---
    # 检查解压目标目录是否存在主要文件，避免重复解压
    # (这里假设解压后会有 ratings.dat)
    potential_ratings_path = os.path.join(temp_dir, extracted_folder_name_pattern, 'ratings.dat')
    # 对于 ml-10m，文件名可能是 ratings.dat 或 ml-10m/ratings.dat
    if dataset_name == 'ml-10m':
         potential_ratings_path = os.path.join(temp_dir, 'ml-10M100K', 'ratings.dat') # ml-10m 特殊处理
    elif dataset_name == 'ml-20m':
         potential_ratings_path = os.path.join(temp_dir, 'ml-20m', 'ratings.csv') # ml-20m 是 csv

    if not os.path.exists(potential_ratings_path):
        print(f"解压目标文件 {potential_ratings_path} 不存在，开始解压...")
        if not extract_zip_file(zip_path, temp_dir):
            return False
    else:
        print(f"目标文件 {potential_ratings_path} 已存在，跳过解压。")


    # --- 3. 处理数据 (格式转换) ---
    ratings_output_csv = os.path.join(final_dir, f'{dataset_name}.csv')
    movies_output_json = os.path.join(final_dir, f'{dataset_name}.item.json') # 统一命名为 .item.json

    if os.path.exists(ratings_output_csv) and os.path.exists(movies_output_json):
        print(f"\n处理后的文件已存在，跳过处理: {final_dir}")
        print("\nMovieLens 数据下载和初步处理完成！")
        print("生成的文件:")
        print(f"  - {os.path.abspath(ratings_output_csv)}")
        print(f"  - {os.path.abspath(movies_output_json)}")
        return True

    print("\n开始处理解压后的数据...")
    os.makedirs(final_dir, exist_ok=True)

    # 定位解压后的 ratings 和 movies 文件
    # 需要根据不同的 MovieLens 版本调整查找逻辑
    ratings_file_path = None
    movies_file_path = None
    extracted_dir_path = temp_dir # 解压根目录

    # 简单模式匹配查找 (可能需要根据实际解压结构调整)
    for root, dirs, files in os.walk(extracted_dir_path):
         # print(f"Scanning: {root}") # Debug: 打印扫描路径
         for file in files:
              # print(f"Found file: {file}") # Debug: 打印找到的文件
              if file.lower() == 'ratings.dat' or file.lower() == 'ratings.csv':
                   ratings_file_path = os.path.join(root, file)
              elif file.lower() == 'movies.dat' or file.lower() == 'movies.csv':
                   movies_file_path = os.path.join(root, file)

    if not ratings_file_path or not os.path.exists(ratings_file_path):
        print(f"错误: 无法在 {extracted_dir_path} 及其子目录中找到 ratings 文件 (ratings.dat 或 ratings.csv)")
        return False
    if not movies_file_path or not os.path.exists(movies_file_path):
        print(f"错误: 无法在 {extracted_dir_path} 及其子目录中找到 movies 文件 (movies.dat 或 movies.csv)")
        return False

    print(f"  -> 找到 ratings 文件: {ratings_file_path}")
    print(f"  -> 找到 movies 文件: {movies_file_path}")

    # --- 处理 Ratings ---
    print("  -> 处理 Ratings...")
    is_csv = ratings_file_path.lower().endswith('.csv')
    delimiter = ',' if is_csv else '::'
    # CSV 文件可能有 header，需要跳过
    skip_header = is_csv

    try:
        with open(ratings_file_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(ratings_output_csv, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            reader = csv.reader(f_in, delimiter=delimiter)

            if skip_header:
                 try: next(reader) # 跳过表头
                 except StopIteration: pass # 文件为空

            for row in tqdm(reader, desc="    Processing ratings"):
                if not row: continue # 跳过空行
                try:
                    if len(row) >= 4:
                        user_id, movie_id, rating, timestamp = row[0], row[1], row[2], row[3]
                        # 统一格式: item_id, user_id, rating, timestamp
                        writer.writerow([movie_id, user_id, rating, timestamp])
                    # else: # Debug: 打印格式不符的行
                    #     print(f"Skipping malformed rating line: {row}")
                except IndexError:
                    # print(f"Skipping malformed rating line (IndexError): {row}")
                    continue
    except Exception as e:
        print(f"处理 ratings 文件失败: {e}")
        if os.path.exists(ratings_output_csv): os.remove(ratings_output_csv)
        return False
    print(f"  -> Ratings CSV 已保存到: {ratings_output_csv}")

    # --- 处理 Movies ---
    print("  -> 处理 Movies...")
    is_csv = movies_file_path.lower().endswith('.csv')
    delimiter = ',' if is_csv else '::'
    skip_header = is_csv
    movies_data = {}

    try:
        with open(movies_file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
            reader = csv.reader(f_in, delimiter=delimiter, quotechar='"') # 处理可能的引号

            if skip_header:
                 try: next(reader)
                 except StopIteration: pass

            for row in tqdm(reader, desc="    Processing movies"):
                 if not row: continue
                 try:
                    if len(row) >= 3:
                        movie_id, title_raw, genres_raw = row[0], row[1], row[2]
                        title = title_raw.strip()
                        genres = genres_raw.strip().split('|') if genres_raw else []

                        # 解析年份
                        year = None
                        title_clean = title
                        match = re.search(r'\((\d{4})\)$', title) # 查找末尾的 (YYYY)
                        if match:
                             year = int(match.group(1))
                             title_clean = title[:match.start()].strip()

                        # 添加 title 和 genres 到 description，如果需要
                        description = f"{title_clean} ({year})" if year else title_clean
                        if genres: description += f". Genres: {', '.join(genres)}"

                        movies_data[movie_id] = {
                            'title': title_clean,
                            'year': year,
                            'genres': genres,
                            'description': description # 添加一个简单的 description
                        }
                    # else:
                    #     print(f"Skipping malformed movie line: {row}")
                 except IndexError:
                     # print(f"Skipping malformed movie line (IndexError): {row}")
                     continue
                 except Exception as inner_e: # 捕获行内处理错误
                     print(f"Error processing movie line {row}: {inner_e}")
                     continue

        with open(movies_output_json, 'w', encoding='utf-8') as f_out:
            json.dump(movies_data, f_out, indent=2, ensure_ascii=False)
        print(f"  -> Movies JSON 已保存到: {movies_output_json}")

    except Exception as e:
        print(f"处理 movies 文件失败: {e}")
        if os.path.exists(movies_output_json): os.remove(movies_output_json)
        return False

    print("\nMovieLens 数据下载和初步处理完成！")
    print("生成的文件:")
    print(f"  - {os.path.abspath(ratings_output_csv)}")
    print(f"  - {os.path.abspath(movies_output_json)}")
    return True


# --- 主程序入口 ---

def main():
    parser = argparse.ArgumentParser(description="下载并初步处理 Amazon 或 MovieLens 数据集。")
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=['amazon', 'movielens'],
        help="数据源类型 ('amazon' 或 'movielens')"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="具体的 数据集名称 (例如 'Musical_Instruments', 'Baby', 'ml-1m', 'ml-20m')"
    )
    parser.add_argument(
        '--data_version',
        type=str,
        default='14', # 默认为 18，仅 Amazon 需要
        choices=['14', '18'],
        help="Amazon 数据集年份版本 (14 或 18)，仅当 source='amazon' 时需要。"
    )
    parser.add_argument('--output_dir', type=str, default='../datasets', help="保存数据的根目录。")
    args = parser.parse_args()

    # 根据数据源调用不同的处理函数
    if args.source == 'amazon':
        process_amazon(args.dataset, args.data_version, args.output_dir)
    elif args.source == 'movielens':
        # 验证 MovieLens dataset name
        valid_ml_datasets = ['ml-1m', 'ml-10m', 'ml-20m']
        if args.dataset not in valid_ml_datasets:
             print(f"错误: 对于 source='movielens'，--dataset 必须是 {valid_ml_datasets} 中的一个。")
             sys.exit(1)
        process_movielens(args.dataset, args.output_dir)
    else:
        # 理论上 argparse 会处理 choices，这里是兜底
        print(f"错误: 不支持的数据源 '{args.source}'")
        sys.exit(1)

if __name__ == '__main__':
    main()