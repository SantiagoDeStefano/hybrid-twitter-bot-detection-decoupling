# preprocess_2.py
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import pandas as pd
import json

MAX_LENGTH = 50
MAX_TWEETS_PER_USER = 20
BATCH_SIZE = int(os.environ.get('EMBED_BATCH_SIZE', '64'))
TWEET_FLUSH_SIZE = int(os.environ.get('TWEET_FLUSH_SIZE', '2048'))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'

user = pd.read_json('E:/Twibot-22/user.json')

user_text = list(user['description'])
with open('processed_data/id_tweet.json', 'r', encoding='utf-8') as f:
    each_user_tweets = json.load(f)

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base').to(DEVICE)
model.eval()


def embed_texts(texts):
    if not texts:
        return torch.empty((0, 768), dtype=torch.float32)

    out = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            if USE_AMP:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    hidden = model(**encoded).last_hidden_state
            else:
                hidden = model(**encoded).last_hidden_state

        pooled = hidden.mean(dim=1).float().cpu()
        out.append(pooled)

    return torch.cat(out, dim=0)

def Des_embbeding():
    print('Running feature1 embedding')
    path = './processed_data/des_tensor.pt'
    if not os.path.exists(path):
        des_tensor = torch.zeros((len(user_text), 768), dtype=torch.float32)
        valid_idx = []
        valid_text = []

        for i, each in enumerate(user_text):
            if isinstance(each, str) and each:
                valid_idx.append(i)
                valid_text.append(each)

        embedded = embed_texts(valid_text)
        if len(valid_idx) > 0:
            des_tensor[torch.tensor(valid_idx, dtype=torch.long)] = embedded

        torch.save(des_tensor, path)
    else:
        des_tensor = torch.load(path)

    print('Finished')
    return des_tensor

def tweets_embedding():
    print('Running feature2 embedding')
    path = './processed_data/tweets_tensor.pt'

    num_users = len(each_user_tweets)
    tweet_sum = torch.zeros((num_users, 768), dtype=torch.float32)
    tweet_count = torch.zeros((num_users,), dtype=torch.float32)

    pending_users = []
    pending_texts = []

    def flush_pending():
        nonlocal pending_users, pending_texts, tweet_sum, tweet_count
        if not pending_texts:
            return

        embedded = embed_texts(pending_texts)
        user_idx = torch.tensor(pending_users, dtype=torch.long)
        tweet_sum.index_add_(0, user_idx, embedded)
        tweet_count.index_add_(0, user_idx, torch.ones(len(pending_users), dtype=torch.float32))
        pending_users = []
        pending_texts = []

    for i in tqdm(range(num_users), desc='users'):
        tweets = each_user_tweets.get(str(i), [])
        if not tweets:
            continue

        for each_tweet in tweets[:MAX_TWEETS_PER_USER]:
            if isinstance(each_tweet, str) and each_tweet:
                pending_users.append(i)
                pending_texts.append(each_tweet)

        if len(pending_texts) >= TWEET_FLUSH_SIZE:
            flush_pending()

    flush_pending()

    tweet_tensor = torch.zeros((num_users, 768), dtype=torch.float32)
    non_zero = tweet_count > 0
    tweet_tensor[non_zero] = tweet_sum[non_zero] / tweet_count[non_zero].unsqueeze(1)
    torch.save(tweet_tensor, './processed_data/tweets_tensor.pt')
    print('Finished')

Des_embbeding()
tweets_embedding()