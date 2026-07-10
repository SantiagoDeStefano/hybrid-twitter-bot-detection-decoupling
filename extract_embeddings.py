"""Extract node embeddings from a trained GNN checkpoint.

Replaces the following near-identical scripts (each differed only in the
model class and which layer(s) were saved):
    extract_embeddings.py                  (BotRGCN, default/last layer)
    extract_embeddings_layer1.py           (BotRGCN, layer 1 only)
    extract_embeddings_alllayers.py        (BotRGCN, concat layer1+layer2)
    extract_embeddings_botgcn.py           (BotGCN, default layer)
    extract_embeddings_alllayers_gcn.py    (BotGCN, all layers)
    extract_embeddings_botgat.py           (BotGAT, default layer)
    extract_embeddings_alllayers_gat.py    (BotGAT, all layers)

Usage:
    python extract_embeddings.py --model botrgcn --layers alllayers --seed 42
    python extract_embeddings.py --model botgat  --layers default   --seed 44
"""
import argparse
import os
import warnings

import torch

from config import PROCESSED_DIR
from Dataset import Twibot22
from model import BotRGCN, BotGCN, BotGAT

warnings.filterwarnings("ignore")

MODEL_CLASSES = {"botrgcn": BotRGCN, "botgcn": BotGCN, "botgat": BotGAT}


def build_model(model_name: str, device: str):
    cls = MODEL_CLASSES[model_name]
    if model_name == "botrgcn":
        return cls(cat_prop_size=3, embedding_dimension=64).to(device)
    return cls(cat_prop_size=3, num_prop_size=5, embedding_dimension=64).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CLASSES), default="botrgcn")
    parser.add_argument("--layers", choices=["default", "layer1", "alllayers"], default="alllayers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dataset = Twibot22(root=PROCESSED_DIR, device=args.device, process=False, save=False)
    des_tensor, tweets_tensor, num_prop, category_prop, \
        edge_index, edge_type, labels, train_idx, val_idx, test_idx = dataset.dataloader()

    model = build_model(args.model, args.device)
    ckpt_path = os.path.join(PROCESSED_DIR, f"{args.model}_best_{args.seed}.pt")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    with torch.no_grad():
        output, embedding = model(
            des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type
        )

    suffix = "" if args.layers == "default" else f"_{args.layers}"
    out_path = os.path.join(PROCESSED_DIR, f"{args.model}{suffix}_embeddings_{args.seed}.pt")
    torch.save(embedding.cpu(), out_path)
    print(f"Seed {args.seed} [{args.model}/{args.layers}] embedding shape: {embedding.shape}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
