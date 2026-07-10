import warnings

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
)

from config import DEFAULT_XGB_PARAMS, SEEDS
from data_utils import load_labels_and_splits, load_embeddings

warnings.filterwarnings("ignore")

d = load_labels_and_splits()
y_train, y_test = d["y_train"], d["y_test"]
scale_pos_weight = d["scale_pos_weight"]


def fit_predict(X_train, X_test, use_weight: bool):
    params = dict(DEFAULT_XGB_PARAMS)
    if use_weight:
        params["scale_pos_weight"] = scale_pos_weight
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train, verbose=False)
    return clf.predict(X_test), clf.predict_proba(X_test)[:, 1]


print("\n[Class Weighting Ablation]")
weighting_results = []
for seed in SEEDS:
    emb = load_embeddings("botrgcn", seed, layers="alllayers")
    X_train, X_test = emb[d["train_mask"]], emb[d["test_mask"]]

    for use_weight in (False, True):
        y_pred, y_prob = fit_predict(X_train, X_test, use_weight)
        weighting_results.append(dict(
            seed=seed, weighted=use_weight,
            f1=f1_score(y_test, y_pred) * 100,
            precision=precision_score(y_test, y_pred) * 100,
            recall=recall_score(y_test, y_pred) * 100,
            roc_auc=roc_auc_score(y_test, y_prob) * 100,
            pr_auc=average_precision_score(y_test, y_prob) * 100,
        ))

for weighted in (False, True):
    rows = [r for r in weighting_results if r["weighted"] == weighted]
    label = "With scale_pos_weight" if weighted else "Without scale_pos_weight"
    f1s = [r["f1"] for r in rows]
    print(f"{label:<30} F1={np.mean(f1s):.2f}±{np.std(f1s):.2f}")

print("\n[Threshold Ablation]")
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
threshold_results = {t: [] for t in thresholds}

for seed in SEEDS:
    emb = load_embeddings("botrgcn", seed, layers="alllayers")
    X_train, X_test = emb[d["train_mask"]], emb[d["test_mask"]]

    params = dict(DEFAULT_XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train, verbose=False)
    y_prob = clf.predict_proba(X_test)[:, 1]

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        threshold_results[t].append(f1_score(y_test, y_pred) * 100)

for t, f1s in threshold_results.items():
    marker = " <- default" if t == 0.5 else ""
    print(f"threshold={t:<5} F1={np.mean(f1s):.2f}{marker}")
