# preprocess_1.py
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime as dt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_column(a):
    arr = np.asarray(a, dtype=np.float32)
    mean = arr.mean() if arr.size else 0.0
    std = arr.std() if arr.size else 0.0
    if std > 0:
        arr = (arr - mean) / std
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    np.nan_to_num(arr, copy=False)
    return arr.reshape(-1, 1)


def main():
    print('loading raw data')
    base = 'E:/TwiBot-22/'
    ensure_dir('./processed_data')

    user = pd.read_json(base + 'user.json')
    user_idx = user['id']
    uid_index = {uid: i for i, uid in enumerate(user_idx.values)}

    print('extracting labels and splits')
    split = pd.read_csv(base + 'split.csv')
    label = pd.read_csv(base + 'label.csv')

    uid_label = dict(zip(label['id'].values, label['label'].values))
    uid_split = dict(zip(split['id'].values, split['split'].values))

    label_new = np.zeros(len(user), dtype=np.int64)
    train_idx = []
    val_idx = []
    test_idx = []

    for i, uid in enumerate(tqdm(user_idx.values, desc='label loop')):
        if uid_label.get(uid, 'human') != 'human':
            label_new[i] = 1
        split_val = uid_split.get(uid, 'val')
        if split_val == 'train':
            train_idx.append(i)
        elif split_val == 'test':
            test_idx.append(i)
        else:
            val_idx.append(i)

    torch.save(torch.from_numpy(label_new), './processed_data/label.pt')
    torch.save(torch.LongTensor(train_idx), './processed_data/train_idx.pt')
    torch.save(torch.LongTensor(val_idx), './processed_data/val_idx.pt')
    torch.save(torch.LongTensor(test_idx), './processed_data/test_idx.pt')

    print('extracting edge_index&edge_type (chunked)')
    edge_chunks = []
    type_chunks = []
    chunk_size = 200_000

    # Create a fast lookup Series for uid to index
    uid_to_idx = pd.Series(range(len(user_idx)), index=user_idx.values, dtype=np.int64)

    for chunk in tqdm(pd.read_csv(base + 'edge.csv', chunksize=chunk_size), desc='edge chunks'):
        chunk = chunk[chunk['relation'].isin(['followers', 'following'])].copy()
        if len(chunk) == 0:
            continue

        # Map source and target to indices using Series map (faster than dict)
        chunk['src'] = chunk['source_id'].map(uid_to_idx)
        chunk['tgt'] = chunk['target_id'].map(uid_to_idx)

        # Drop rows where src or tgt is NaN (invalid uids)
        chunk.dropna(subset=['src', 'tgt'], inplace=True)

        if len(chunk) == 0:
            continue

        # Convert to int64
        chunk['src'] = chunk['src'].astype(np.int64)
        chunk['tgt'] = chunk['tgt'].astype(np.int64)
        chunk['etype'] = chunk['relation'].map({'followers': 0, 'following': 1}).astype(np.int64)

        # Create tensors
        ei = torch.tensor(np.vstack((chunk['src'].values, chunk['tgt'].values)), dtype=torch.long)
        et = torch.tensor(chunk['etype'].values, dtype=torch.long)
        edge_chunks.append(ei)
        type_chunks.append(et)

    if edge_chunks:
        edge_index = torch.cat(edge_chunks, dim=1)
        edge_type = torch.cat(type_chunks, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    torch.save(edge_index, './processed_data/edge_index.pt')
    torch.save(edge_type, './processed_data/edge_type.pt')

    print('extracting num_properties')
    public_metrics = user['public_metrics']

    following_count = [x['following_count'] for x in public_metrics]
    followers_count = [x['followers_count'] for x in public_metrics]
    tweet_count = [x['tweet_count'] for x in public_metrics]
    screen_name_length = [len(x) for x in user['name']]

    created_at = pd.to_datetime(user['created_at'], unit='s')
    date0 = dt.strptime('Tue Sep 5 00:00:00 +0000 2020 ', '%a %b %d %X %z %Y ')
    active_days = [(date0 - x).days for x in created_at]

    # normalize
    arr_followers = normalize_column(followers_count)
    arr_active = normalize_column(active_days)
    arr_screen = normalize_column(screen_name_length)
    arr_following = normalize_column(following_count)
    arr_tweets = normalize_column(tweet_count)

    num_properties_tensor = torch.from_numpy(np.concatenate([
        arr_followers,
        arr_active,
        arr_screen,
        arr_following,
        arr_tweets,
    ], axis=1))

    torch.save(num_properties_tensor, './processed_data/num_properties_tensor.pt')

    print('extracting cat_properties')
    protected = np.array([1.0 if x is True else 0.0 for x in user['protected']], dtype=np.float32)
    verified = np.array([1.0 if x is True else 0.0 for x in user['verified']], dtype=np.float32)
    default_profile_image = np.array([
        1.0 if (not isinstance(x, str) or x == '' or x == 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png') else 0.0
        for x in user['profile_image_url']
    ], dtype=np.float32)

    cat_properties_tensor = torch.from_numpy(np.vstack([protected, verified, default_profile_image]).T)
    torch.save(cat_properties_tensor, './processed_data/cat_properties_tensor.pt')

    print("extracting each_user's tweets (streaming)")
    id_tweet = {i: [] for i in range(len(user_idx))}

    for i in range(9):
        tweet_file = Path(base) / f'tweet_{i}.json'
        if not tweet_file.exists():
            continue

        with tweet_file.open('r', encoding='utf-8') as f:
            first_char = f.read(1)
            if not first_char:
                continue
            f.seek(0)

            if first_char == '[':
                try:
                    import ijson
                    tweet_iter = ijson.items(f, 'item')
                except (ImportError, Exception):
                    f.seek(0)
                    tweet_iter = (json.loads(line) for line in f if line.strip())
            else:
                tweet_iter = (json.loads(line) for line in f if line.strip())

            for tw in tqdm(tweet_iter, desc=f'tweet_{i}'):
                uid = 'u' + str(tw.get('author_id', ''))
                idx = uid_index.get(uid)
                if idx is None:
                    continue

                id_tweet[idx].append(tw.get('text', ''))

    with open('./processed_data/id_tweet.json', 'w', encoding='utf-8') as f:
        json.dump(id_tweet, f)


if __name__ == '__main__':
    main()