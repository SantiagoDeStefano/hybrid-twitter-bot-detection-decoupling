import os

DATA_ROOT = os.environ.get("TWIBOT22_ROOT", "./TwiBot-22")

PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "./processed_data")

DEFAULT_XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    device="cuda",
)

SEEDS = range(42, 47)
