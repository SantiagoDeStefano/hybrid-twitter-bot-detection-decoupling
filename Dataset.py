import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset


class Twibot22(Dataset):
    def __init__(self, root='./processed_data/', device='cpu', process=False, save=False):
        self.root = root
        self.device = device
        self.root_path = Path(root)
        script_dir = Path(__file__).resolve().parent
        parent_dir = script_dir.parent
        self.search_roots = []
        for base in (self.root_path, script_dir / 'processed_data', parent_dir / 'processed_data'):
            if base not in self.search_roots:
                self.search_roots.append(base)
        # process=False for TwiBot-22: all tensors are pre-built by preprocess_1/2.py
        # process=True is only used for the legacy TwiBot-20 workflow (not needed here)

    def _p(self, filename):
        """Return full path for a processed file, searching known data dirs."""
        for base in self.search_roots:
            candidate = base / filename
            if candidate.exists():
                return candidate
        return self.root_path / filename

    # ── loaders ───────────────────────────────────────────────────────────────
    def load_labels(self):
        print('Loading labels...', end='   ')
        labels = torch.load(str(self._p('label.pt'))).to(self.device)
        print('Finished')
        return labels

    def Des_embbeding(self):
        print('Loading description embeddings...', end='   ')
        path = self._p('des_tensor.pt')
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run preprocess_2.py first.")
        des_tensor = torch.load(str(path)).to(self.device)
        print('Finished')
        return des_tensor

    def tweets_embedding(self):
        print('Loading tweet embeddings...', end='   ')
        path = self._p('tweets_tensor.pt')
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run preprocess_2.py first.")
        tweets_tensor = torch.load(str(path)).to(self.device)
        print('Finished')
        return tweets_tensor

    def num_prop_preprocess(self):
        print('Loading num_properties...', end='   ')
        path = self._p('num_properties_tensor.pt')
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run preprocess_1.py first.")
        num_prop = torch.load(str(path)).to(self.device)
        print('Finished')
        return num_prop

    def cat_prop_preprocess(self):
        print('Loading cat_properties...', end='   ')
        path = self._p('cat_properties_tensor.pt')
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run preprocess_1.py first.")
        cat_prop = torch.load(str(path)).to(self.device)
        print('Finished')
        return cat_prop

    def Build_Graph(self):
        print('Loading graph...', end='   ')
        ei_path = self._p('edge_index.pt')
        et_path = self._p('edge_type.pt')
        if not ei_path.exists() or not et_path.exists():
            raise FileNotFoundError(
                "Missing edge_index.pt or edge_type.pt. Run preprocess_1.py first.")
        edge_index = torch.load(str(ei_path)).to(self.device)
        edge_type  = torch.load(str(et_path)).to(self.device)
        print('Finished')
        return edge_index, edge_type

    def train_val_test_mask(self):
        train_idx = torch.load(str(self._p('train_idx.pt')))
        val_idx   = torch.load(str(self._p('val_idx.pt')))
        test_idx  = torch.load(str(self._p('test_idx.pt')))
        return train_idx, val_idx, test_idx

    def dataloader(self):
        labels                          = self.load_labels()
        des_tensor                      = self.Des_embbeding()
        tweets_tensor                   = self.tweets_embedding()
        num_prop                        = self.num_prop_preprocess()
        category_prop                   = self.cat_prop_preprocess()
        edge_index, edge_type           = self.Build_Graph()
        train_idx, val_idx, test_idx    = self.train_val_test_mask()
        return (des_tensor, tweets_tensor, num_prop, category_prop,
                edge_index, edge_type, labels, train_idx, val_idx, test_idx)