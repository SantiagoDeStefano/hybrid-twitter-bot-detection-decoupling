import json
import os
import warnings

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from config import DEFAULT_XGB_PARAMS, PROCESSED_DIR
from data_utils import load_labels_and_splits, load_embeddings

warnings.filterwarnings("ignore")

PARAM_LABELS = {
    "n_estimators": "Number of Trees",
    "max_depth": "Max Depth",
    "learning_rate": "Learning Rate",
    "subsample": "Subsample",
    "colsample_bytree": "Colsample by Tree",
    "reg_alpha": "L1 Regularization (\u03b1)",
    "reg_lambda": "L2 Regularization (\u03bb)",
}
DEFAULT_VALS = {
    "n_estimators": "300", "max_depth": "6", "learning_rate": "0.1",
    "subsample": "0.8", "colsample_bytree": "0.8", "reg_alpha": "0", "reg_lambda": "1.0",
}


def plot_hyperparam_sweeps():
    with open(os.path.join(PROCESSED_DIR, "hyperparam_sweep_results.json")) as f:
        results = json.load(f)

    for param_name, param_results in results.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        values = [r["value"] for r in param_results]
        f1s = [r["f1_mean"] for r in param_results]
        stds = [r["f1_std"] for r in param_results]
        default = DEFAULT_VALS[param_name]

        ax.errorbar(range(len(values)), f1s, yerr=stds, fmt="b-o", linewidth=2,
                    markersize=6, capsize=4, elinewidth=1.5, label="Mean \u00b1 Std")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], rotation=30, ha="right")

        str_values = [str(v) for v in values]
        if default in str_values:
            ax.axvline(x=str_values.index(default), color="red", linestyle="--",
                       alpha=0.7, label="Default")

        ax.set_title(PARAM_LABELS[param_name], fontsize=11, fontweight="bold")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(min(f1s) - max(stds) - 0.5, max(f1s) + max(stds) + 0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.suptitle("XGBoost Hyperparameter Sensitivity", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(PROCESSED_DIR, f"hyperparam_{param_name}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


def plot_roc_pr(seed: int = 42):
    d = load_labels_and_splits()
    y_train, y_test = d["y_train"], d["y_test"]

    emb = load_embeddings("botrgcn", seed, layers="alllayers")
    X_train, X_test = emb[d["train_mask"]], emb[d["test_mask"]]

    clf = xgb.XGBClassifier(**dict(DEFAULT_XGB_PARAMS, scale_pos_weight=d["scale_pos_weight"]))
    clf.fit(X_train, y_train, verbose=False)
    y_prob = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    baseline_precision = y_test.sum() / len(y_test)

    print(f"ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}  baseline: {baseline_precision:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"Two-Stage XGBoost (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.suptitle("Two-Stage BotRGCN + XGBoost", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, "r-", linewidth=2, label=f"Two-Stage XGBoost (AUC={pr_auc:.4f})")
    ax.axhline(y=baseline_precision, color="k", linestyle="--", linewidth=1,
               label=f"Random baseline ({baseline_precision:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.suptitle("Two-Stage BotRGCN + XGBoost", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, "pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved roc_curve.png, pr_curve.png")


if __name__ == "__main__":
    plot_hyperparam_sweeps()
    plot_roc_pr()
