import warnings

import numpy as np
import torch
from tqdm import tqdm

from config import SEEDS
from data_utils import load_labels_and_splits, load_embeddings
from mlp_utils import train_mlp

warnings.filterwarnings("ignore")

ARCHITECTURES = {
    "MLP-1L (128)": (128,),
    "MLP-2L (256-64)": (256, 64),
    "MLP-2L (512-256)": (512, 256),
    "MLP-3L (512-256-64)": (512, 256, 64),
}
LR = 1e-2
DROPOUT = 0.3


def main():
    d = load_labels_and_splits()
    y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]

    emb_42 = load_embeddings("botrgcn", 42, layers="alllayers")
    X_train = emb_42[d["train_mask"]]
    X_val = emb_42[d["val_mask"]]
    X_test = emb_42[d["test_mask"]]

    print("\n[Phase 1: Architecture search on seed 42]")
    best = {"f1": 0}
    for name, hidden in tqdm(ARCHITECTURES.items(), desc="Phase 1"):
        torch.manual_seed(42)
        f1 = train_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
                        hidden_layers=hidden, lr=LR, dropout=DROPOUT)
        print(f"  {name:<25} F1={f1:.2f}")
        if f1 > best["f1"]:
            best = {"f1": f1, "arch": name, "hidden": hidden}

    print(f"\nBest config: {best['arch']} (F1={best['f1']:.2f})")

    print("\n[Phase 2: Best config across all seeds]")
    seed_f1s = []
    for seed in tqdm(SEEDS, desc="Phase 2"):
        emb = load_embeddings("botrgcn", seed, layers="alllayers")
        X_tr, X_v, X_te = emb[d["train_mask"]], emb[d["val_mask"]], emb[d["test_mask"]]
        torch.manual_seed(42)
        f1 = train_mlp(X_tr, y_train, X_v, y_val, X_te, y_test,
                        hidden_layers=best["hidden"], lr=LR, dropout=DROPOUT)
        seed_f1s.append(f1)
        print(f"  Seed {seed}: F1={f1:.2f}")

    print(f"\nBest Tuned MLP: {np.mean(seed_f1s):.2f} ± {np.std(seed_f1s):.2f}")


if __name__ == "__main__":
    main()
