import argparse
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import torch
import xgboost as xgb

from config import DEFAULT_XGB_PARAMS
from data_utils import load_labels_and_splits, load_embeddings
from mlp_utils import train_mlp

warnings.filterwarnings("ignore")


def report(name, y_test, y_pred):
    print(f"  {name:<20} F1={f1_score(y_test, y_pred)*100:.2f} "
          f"P={precision_score(y_test, y_pred)*100:.2f} "
          f"R={recall_score(y_test, y_pred)*100:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="botrgcn")
    args = parser.parse_args()

    d = load_labels_and_splits()
    y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]
    scale_pos_weight = d["scale_pos_weight"]

    emb = load_embeddings(args.model, args.seed, layers="alllayers")
    X_train = emb[d["train_mask"]]
    X_val = emb[d["val_mask"]]
    X_test = emb[d["test_mask"]]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"\n{'='*60}\nSEED {args.seed} — {args.model} embeddings\n{'='*60}")

    clf_xgb = xgb.XGBClassifier(**dict(DEFAULT_XGB_PARAMS, scale_pos_weight=scale_pos_weight))
    clf_xgb.fit(X_train, y_train, verbose=False)
    report("XGBoost", y_test, clf_xgb.predict(X_test))

    clf_lr = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced", n_jobs=-1)
    clf_lr.fit(X_train_s, y_train)
    report("LogisticRegression", y_test, clf_lr.predict(X_test_s))

    clf_rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    clf_rf.fit(X_train, y_train)
    report("RandomForest", y_test, clf_rf.predict(X_test))

    torch.manual_seed(42)
    mlp_preds = train_mlp(X_train, y_train, X_val, d["y_val"], X_test, return_preds=True)
    report("MLP (tuned)", y_test, mlp_preds)


if __name__ == "__main__":
    main()
