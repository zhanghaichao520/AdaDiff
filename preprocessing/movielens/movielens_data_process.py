import argparse
import collections
import json
import os
import random
import torch
from tqdm import tqdm
import numpy as np
from ..utils import check_path, clean_text, write_json_file, write_remap_index, load_json

def load_ratings(file):
    """加载评分数据"""
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                users.add(user)
                items.add(item)
                inters.add((user, item, float(rating), int(time)))
            except ValueError:
                continue
    return users, items, inters

def load_movie_metadata(file_path):
    """加载电影元数据"""
    movies = {}
    with open(file_path, 'r', encoding='utf-8') as fp:
        movies_data = json.load(fp)
        
    for movie_id, data in movies_data.items():
        movies[movie_id] = {
            'title': clean_text(data.get('title', '')),
            'description': clean_text(data.get('description', '')),
            'genres': data.get('genres', []),
            'year': data.get('year', None)
        }
    return movies

def preprocess_rating(args):
    """预处理评分数据"""
    print('Process rating data: ')
    print(' Dataset: ', args.dataset)

    # 构建输入文件路径
    input_root_path = os.path.join(args.input_path, args.dataset, 'processed')
    
    # 加载评分数据
    rating_file_path = os.path.join(input_root_path, f'{args.dataset}.csv')
    if not os.path.exists(rating_file_path):
        raise FileNotFoundError(f"Ratings file not found: {rating_file_path}")
    
    _, _, rating_inters = load_ratings(rating_file_path)

    # 加载电影元数据
    movies_file_path = os.path.join(input_root_path, f'{args.dataset}_movies.json')
    if not os.path.exists(movies_file_path):
        raise FileNotFoundError(f"Movies file not found: {movies_file_path}")
    
    movie_items = load_movie_metadata(movies_file_path)

    print('The number of raw inters: ', len(rating_inters))
    rating_inters = make_inters_in_order(rating_inters)
    
    # 过滤掉没有元数据的交互
    filtered_inters = []
    for inter in tqdm(rating_inters, desc="Filtering interactions by movie metadata"):
        if inter[1] in movie_items:
            filtered_inters.append(inter)
    rating_inters = filtered_inters
    print(f"Interactions after metadata filtering: {len(rating_inters)}")

    # K-core 过滤
    rating_inters = filter_inters(rating_inters, can_items=None,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)
    
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')
    return rating_inters, movie_items

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
        print('\nFiltering by movie metadata: ')
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
    """
    将原始交互映射为 user2items, user2index, item2index。
    """
    # 1) 收集全集
    all_users = {u for (u, i, r, t) in inters}
    all_items = {i for (u, i, r, t) in inters}

    # 2) 稳定排序并建立映射
    users_sorted = sorted(all_users)
    items_sorted = sorted(all_items)
    user2index = {u: idx for idx, u in enumerate(users_sorted)}
    item2index = {i: idx for idx, i in enumerate(items_sorted)}

    # 3) 逐条交互按时间原序写回（不打乱时间）
    user2items = collections.defaultdict(list)
    for u, it, r, ts in inters:
        uid = user2index[u]
        iid = item2index[it]
        user2items[uid].append(iid)

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

def convert_to_atomic_files(args, train_data, valid_data, test_data, max_history_len=50, use_sliding_window=True):
    """
    保存为 JSONL 文件，每行包含三个字段：user, history, target
    """
    print('Convert dataset to JSONL:')
    print(' Dataset: ', args.dataset)

    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    output_dir = os.path.join(args.output_path, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # --- Train ---
    train_path = os.path.join(output_dir, f"{args.dataset}.train.jsonl")
    with open(train_path, 'w') as f:
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)

            if use_sliding_window:
                # 滑动窗口：逐步生成 prefix → target
                for target_idx in range(1, seq_len):
                    target_item = item_seq[-target_idx]
                    seq = item_seq[:-target_idx][-max_history_len:]
                    json.dump({"user": uid, "history": seq, "target": target_item}, f)
                    f.write("\n")
            else:
                # 只取最后一个
                target_item = item_seq[-1]
                seq = item_seq[:-1][-max_history_len:]
                json.dump({"user": uid, "history": seq, "target": target_item}, f)
                f.write("\n")

    #  Valid 
    valid_path = os.path.join(output_dir, f"{args.dataset}.valid.jsonl")
    with open(valid_path, 'w') as f:
        for uid in uid_list:
            item_seq = train_data[uid][-max_history_len:]
            target_item = valid_data[uid][0]
            json.dump({"user": uid, "history": item_seq, "target": target_item}, f)
            f.write("\n")

    #  Test 
    test_path = os.path.join(output_dir, f"{args.dataset}.test.jsonl")
    with open(test_path, 'w') as f:
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-max_history_len:]
            target_item = test_data[uid][0]
            json.dump({"user": uid, "history": item_seq, "target": target_item}, f)
            f.write("\n")

    print(f"JSONL files saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='e.g., ml-1m, ml-10m, ml-20m')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='../datasets', help='Root path containing MovieLens datasets')
    parser.add_argument('--output_path', type=str, default='../datasets')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print('\n' + '=' * 20)
    print(f"Start processing dataset: {args.dataset}")
    print('=' * 20 + '\n')
        
    rating_inters, movie_items = preprocess_rating(args)
    
    all_inters, train_inters, valid_inters, test_inters, user2index, item2index = generate_data(args, rating_inters)
    
    output_dataset_path = os.path.join(args.output_path, args.dataset)
    check_path(output_dataset_path)

    write_json_file(all_inters, os.path.join(output_dataset_path, f'{args.dataset}.inter.json'))
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

    item2feature = collections.defaultdict(dict)
    for item, item_id in item2index.items():
        # 确保item在 movie_items 
        if item in movie_items:
            item2feature[item_id] = movie_items[item]

    print("user:", len(user2index))
    print("item:", len(item2index))

    write_json_file(item2feature, os.path.join(output_dataset_path, f'{args.dataset}.item.json'))
    write_remap_index(user2index, os.path.join(output_dataset_path, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(output_dataset_path, f'{args.dataset}.item2id'))
    
    print(f"\nFinished processing dataset: {args.dataset}")
