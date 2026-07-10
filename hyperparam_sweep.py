import json
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

from config import DEFAULT_XGB_PARAMS, PROCESSED_DIR, SEEDS
from data_utils import load_labels_and_splits, load_embeddings

warnings.filterwarnings("ignore")

PARAM_GRIDS = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [3, 4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1.0, 10.0],
    "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
}


def main():
    d = load_labels_and_splits()
    y_train, y_test = d["y_train"], d["y_test"]
    base_params = dict(DEFAULT_XGB_PARAMS, scale_pos_weight=d["scale_pos_weight"])

    all_results = {p: {str(v): [] for v in vals} for p, vals in PARAM_GRIDS.items()}

    for seed in SEEDS:
        print(f"\n{'='*20} SEED {seed} {'='*20}")
        emb = load_embeddings("botrgcn", seed, layers="alllayers")
        X_train, X_test = emb[d["train_mask"]], emb[d["test_mask"]]

        for param_name, values in PARAM_GRIDS.items():
            for val in values:
                params = dict(base_params, **{param_name: val})
                clf = xgb.XGBClassifier(**params)
                clf.fit(X_train, y_train, verbose=False)
                f1 = f1_score(y_test, clf.predict(X_test)) * 100
                all_results[param_name][str(val)].append(f1)

    results = {
        p: [{"value": v, "f1_mean": np.mean(f1s), "f1_std": np.std(f1s)} for v, f1s in vd.items()]
        for p, vd in all_results.items()
    }

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "hyperparam_sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    rows = [{"parameter": p, **r} for p, rs in results.items() for r in rs]
    pd.DataFrame(rows).to_csv(os.path.join(PROCESSED_DIR, "hyperparam_sweep_results.csv"), index=False)
    print("\nSaved hyperparam_sweep_results.{json,csv}")


if __name__ == "__main__":
    main()
