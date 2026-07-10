import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import DATA_ROOT, PROCESSED_DIR


def load_labels_and_splits(data_root: str = DATA_ROOT):
    user = pd.read_json(os.path.join(data_root, "user.json"))
    user_order = user["id"].tolist()

    label_df = pd.read_csv(os.path.join(data_root, "label.csv"))
    uid_label = dict(zip(label_df["id"], label_df["label"]))

    split_df = pd.read_csv(os.path.join(data_root, "split.csv"))
    uid_split = dict(zip(split_df["id"], split_df["split"]))

    labels, splits = [], []
    for uid in tqdm(user_order, desc="Loading labels/splits"):
        labels.append(1 if uid_label.get(uid, "human") == "bot" else 0)
        splits.append(uid_split.get(uid, "val"))

    labels = np.array(labels)
    splits = np.array(splits)

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    y_train, y_val, y_test = labels[train_mask], labels[val_mask], labels[test_mask]
    neg, pos = np.bincount(y_train)

    return dict(
        labels=labels,
        splits=splits,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scale_pos_weight=neg / pos,
    )


def load_embeddings(model_name: str, seed: int, layers: str = "alllayers",
                     processed_dir: str = PROCESSED_DIR):
    """Load a saved embedding tensor as a numpy array.

    model_name: 'botrgcn' | 'botgcn' | 'botgat'
    layers: 'alllayers' | 'layer1' | 'default'
    """
    suffix = "" if layers == "default" else f"_{layers}"
    path = os.path.join(processed_dir, f"{model_name}{suffix}_embeddings_{seed}.pt")
    return torch.load(path).numpy()
