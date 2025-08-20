from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from src.config import DATA_DIR, ARTIFACTS_DIR, ENCODER_PATH, PERSONA_MODEL_PATH
from src.features.persona_clustering import load as load_persona, assign_personas
from src.features.preprocess import select_user_features

OUT_PATH = ARTIFACTS_DIR / "ltr_model.joblib"
RANDOM_STATE = 42

# Feature schema (shared with API via src/models/ltr.py)
NUM = ["age", "baseline_activity_min_per_day", "duration_min", "day_of_week", "popularity"]
CAT = ["premium", "push_opt_in", "chronotype", "primary_goal", "type",
       "intensity", "difficulty", "goal_tag", "hour_bucket", "persona"]

def build_dataset():
    users = pd.read_csv(DATA_DIR / "users.csv")
    content = pd.read_csv(DATA_DIR / "content_catalog.csv")
    inter = pd.read_csv(DATA_DIR / "interactions.csv")

    # popularity prior (CTR per content)
    pop = inter.groupby("content_id")["reward"].mean().rename("popularity").reset_index()
    content = content.merge(pop, on="content_id", how="left")
    content["popularity"] = content["popularity"].fillna(0.0)

    # personas for users
    pre, km = load_persona(ENCODER_PATH, PERSONA_MODEL_PATH)
    users_persona = assign_personas(select_user_features(users), pre, km)[["user_id", "persona"]]

    # join
    df = inter.merge(users, on="user_id", how="left")
    df = df.merge(content, on="content_id", how="left")
    df = df.merge(users_persona, on="user_id", how="left", suffixes=("", "_p"))

    # target
    y = df["reward"].astype(int).values

    # features
    X = df[["age","baseline_activity_min_per_day","duration_min","day_of_week",
            "premium","push_opt_in","chronotype","primary_goal","type","intensity",
            "difficulty","goal_tag","hour_bucket","persona","popularity"]].copy()

    # dtype hygiene
    X["premium"] = X["premium"].astype(bool)
    X["push_opt_in"] = X["push_opt_in"].astype(bool)
    X["persona"] = X["persona"].astype(str)  # categorical

    return X, y

def choose_estimator(y):
    try:
        import xgboost as xgb
        pos = np.sum(y == 1); neg = np.sum(y == 0)
        spw = float(neg) / float(max(pos, 1))
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=4,
            random_state=RANDOM_STATE,
            scale_pos_weight=spw,
        )
        return clf, "xgboost"
    except Exception:
        clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
        return clf, "logreg"

def train():
    X, y = build_dataset()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])

    clf, name = choose_estimator(ytr)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)

    # quick validation metrics
    p = pipe.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(yva, p)
    ap  = average_precision_score(yva, p)
    ll  = log_loss(yva, p, labels=[0,1])
    print(f"[ltr] model={name}  ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}  logloss={ll:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT_PATH)
    print(f"Saved learned scorer → {OUT_PATH}")

if __name__ == "__main__":
    train()
