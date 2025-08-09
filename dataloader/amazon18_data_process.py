import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
import ast # <-- 引入 ast 模块
from utils import check_path, clean_text, write_json_file, write_remap_index, load_json

# amazon18_dataset2fullname 这个字典不再是必需的，但为了utils中其他可能的使用而保留
from utils import amazon18_dataset2fullname 

def load_ratings(file, images_info):
    # 这个函数假设 images_info 已经加载，对于 2014 数据可能为空，但不影响逻辑
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                # 对于2014数据，images_info可能是空的，这里的逻辑需要调整
                # 简单起见，我们假设只要评分文件里有，就是有效交互
                users.add(user)
                items.add(item)
                inters.add((user, item, float(rating), int(time)))
            except ValueError:
                # print(line)
                continue
    return users, items, inters

def load_meta_items(file_path, data_version):
    """
    加载元数据文件，智能兼容 2014 和 2018 年的数据集格式。
    无论输入是哪个版本，都输出统一格式的字典。
    """
    items = {}
    parser = ast.literal_eval if data_version == '14' else json.loads

    with gzip.open(file_path, "rt", encoding='utf-8') as fp:
        for line in tqdm(fp, desc="Load metas"):
            try:
                data = parser(line)
                item_id = data.get("asin")
                if not item_id:
                    continue

                # 定义统一的数据结构
                unified_info = {
                    "title": "", 
                    "description": "", 
                    "brand": "", 
                    "categories": ""
                }

                # --- 兼容性逻辑 ---
                if data_version == '18':
                    unified_info['title'] = clean_text(data.get('title', ''))
                    unified_info['description'] = clean_text(' '.join(data.get('description', [])))
                    unified_info['brand'] = data.get('brand', '').replace("by\n", "").strip()
                    # 2018版的'category'是列表
                    cats = [cat.strip() for cat in data.get('category', []) if "</span>" not in cat]
                    unified_info['categories'] = ",".join(cats).strip()
                
                elif data_version == '14':
                    unified_info['title'] = clean_text(data.get('title', ''))
                    unified_info['description'] = clean_text(data.get('description', ''))
                    unified_info['brand'] = data.get('brand', '').replace("by\n", "").strip()
                    # 2014版的'categories'是列表的列表
                    cats_list = data.get('categories', [[]])
                    if cats_list and isinstance(cats_list, list) and len(cats_list) > 0:
                        cats = [cat.strip() for cat in cats_list[0] if "</span>" not in cat]
                        unified_info['categories'] = ",".join(cats).strip()
                
                items[item_id] = unified_info
            
            except (ValueError, SyntaxError, TypeError, KeyError):
                continue
    return items

def preprocess_rating(args):
    """
    动态构造文件路径并处理数据，兼容 14 和 18 版本。
    """
    print('Process rating data: ')
    print(' Dataset: ', args.dataset)
    print(' Data Version: ', args.data_version)

    # 动态构造输入文件根目录
    input_root_path = os.path.join(args.input_path, f'amazon{args.data_version}')

    # 加载图片信息 (主要用于2018数据过滤，对于2014数据可能是空文件，但不影响)
    images_info_file = os.path.join(input_root_path, 'Images', f'{args.dataset}_images_info.json')
    if os.path.exists(images_info_file):
        images_info = load_json(images_info_file)
    else:
        # 如果文件不存在（例如对于2014数据），创建一个空字典
        images_info = {}
        print(f"Warning: Image info file not found at {images_info_file}. Proceeding without it.")

    # 动态构造 ratings 文件路径
    rating_file_path = os.path.join(input_root_path, 'Ratings', f'{args.dataset}.csv')
    if not os.path.exists(rating_file_path):
        raise FileNotFoundError(f"Ratings file not found: {rating_file_path}")
    
    # load_ratings 不再需要 images_info 进行核心过滤
    _, _, rating_inters = load_ratings(rating_file_path, images_info)

    # 动态构造 meta 文件路径
    meta_file_path = os.path.join(input_root_path, 'Metadata', f'meta_{args.dataset}.json.gz')
    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_file_path}")

    # 调用兼容版的 load_meta_items
    meta_items = load_meta_items(meta_file_path, args.data_version)

    print('The number of raw inters: ', len(rating_inters))
    rating_inters = make_inters_in_order(rating_inters)
    
    # 过滤掉没有元数据的交互
    filtered_inters = []
    for inter in tqdm(rating_inters, desc="Filtering interactions by meta items"):
        if inter[1] in meta_items:
            filtered_inters.append(inter)
    rating_inters = filtered_inters
    print(f"Interactions after meta filtering: {len(rating_inters)}")

    # K-core 过滤
    rating_inters = filter_inters(rating_inters, can_items=None, # meta过滤已完成
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)
    
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')
    return rating_inters, meta_items

# --- The rest of the script remains mostly the same ---
# (get_user2count, get_item2count, generate_candidates, filter_inters, make_inters_in_order,
# convert_inters2dict, generate_data, convert_to_atomic_files functions are unchanged)
def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count
def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count
def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)
def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []
    if can_items:
        print('\nFiltering by meta items: ')
        for unit in tqdm(inters):
            if unit[1] in can_items.keys():
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)
        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters
def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in tqdm(inters):
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in tqdm(user2inters):
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        interacted_item = set()
        for inter in user_inters:
            if inter[1] in interacted_item:
                continue
            interacted_item.add(inter[1])
            new_inters.append(inter)
    return new_inters
def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index: user2index[user] = len(user2index)
        if item not in item2index: item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
    return user2items, user2index, item2index
def generate_data(args, rating_inters):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters = dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
        valid_inters[u_index] = [str(inters[-2])]
        test_inters[u_index] = [str(inters[-1])]
        assert len(user2items[u_index]) == len(train_inters[u_index]) + len(valid_inters[u_index]) + len(test_inters[u_index])
    return user2items, train_inters, valid_inters, test_inters, user2index, item2index

def convert_to_atomic_files(args, train_data, valid_data, test_data):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    # 定义并创建输出子目录
    training_path = os.path.join(args.output_path, args.dataset, 'training')
    evaluation_path = os.path.join(args.output_path, args.dataset, 'evaluation')
    testing_path = os.path.join(args.output_path, args.dataset, 'testing')

    os.makedirs(training_path, exist_ok=True)
    os.makedirs(evaluation_path, exist_ok=True)
    os.makedirs(testing_path, exist_ok=True)

    # 保存 train.inter 到 training 文件夹
    with open(os.path.join(training_path, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')
    
    # 保存 valid.inter 到 evaluation 文件夹
    with open(os.path.join(evaluation_path, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    # 保存 test.inter 到 testing 文件夹
    with open(os.path.join(testing_path, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIED: Added data_version argument
    parser.add_argument('--data_version', type=str, default='14', choices=['14', '18'], help='Dataset version (14 or 18)')
    parser.add_argument('--dataset', type=str, default='Home', help='e.g., Home, Baby, All_Beauty, etc.')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    # MODIFIED: Changed default path to be more generic
    parser.add_argument('--input_path', type=str, default='../datasets', help='Root path containing amazon14/ and amazon18/ dirs')
    parser.add_argument('--output_path', type=str, default='../datasets')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print('\n' + '=' * 20)
    print(f"Start processing dataset: {args.dataset} (Version: {args.data_version})")
    print('=' * 20 + '\n')
        
    rating_inters, meta_items = preprocess_rating(args)
    
    all_inters, train_inters, valid_inters, test_inters, user2index, item2index = generate_data(args, rating_inters)
    
    output_dataset_path = os.path.join(args.output_path, args.dataset)
    check_path(output_dataset_path)

    write_json_file(all_inters, os.path.join(output_dataset_path, f'{args.dataset}.inter.json'))
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

    item2feature = collections.defaultdict(dict)
    for item, item_id in item2index.items():
        # Ensure item from interactions exists in meta_items before assignment
        if item in meta_items:
            item2feature[item_id] = meta_items[item]

    print("user:", len(user2index))
    print("item:", len(item2index))

    write_json_file(item2feature, os.path.join(output_dataset_path, f'{args.dataset}.item.json'))
    write_remap_index(user2index, os.path.join(output_dataset_path, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(output_dataset_path, f'{args.dataset}.item2id'))
    
    print(f"\nFinished processing dataset: {args.dataset}")