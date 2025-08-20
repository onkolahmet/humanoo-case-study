from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import src.config as cfg
from src.features.preprocess import build_user_preprocessor, select_user_features
from src.features.persona_clustering import fit_kmeans_personas, save as save_persona

def _ensure_kmeans(obj):
    if isinstance(obj, tuple):
        for part in obj:
            if hasattr(part, "predict"):
                return part
        raise AssertionError("fit_kmeans_personas returned a tuple without a KMeans-like object.")
    return obj

def _seed_eval_env(root: Path):
    data = root / "data"
    arts = root / "artifacts"
    data.mkdir(parents=True, exist_ok=True)
    arts.mkdir(parents=True, exist_ok=True)

    users = pd.DataFrame([
        {"user_id":"u1","age":29,"gender":"female","work_pattern":"9-5","primary_goal":"stress",
         "baseline_activity_min_per_day":12,"premium":False,"push_opt_in":True,"chronotype":"morning","language":"en"},
        {"user_id":"u2","age":41,"gender":"male","work_pattern":"shift","primary_goal":"fitness",
         "baseline_activity_min_per_day":3,"premium":True,"push_opt_in":False,"chronotype":"evening","language":"de"},
        {"user_id":"u3","age":35,"gender":"other","work_pattern":"flex","primary_goal":"fitness",
         "baseline_activity_min_per_day":20,"premium":False,"push_opt_in":True,"chronotype":"morning","language":"en"},
    ])
    content = pd.DataFrame([
        {"content_id":"c1","type":"meditation","duration_min":10,"intensity":"low","goal_tag":"stress","difficulty":"beginner"},
        {"content_id":"c2","type":"hiit","duration_min":25,"intensity":"high","goal_tag":"fitness","difficulty":"advanced"},
        {"content_id":"c3","type":"walk","duration_min":20,"intensity":"medium","goal_tag":"fitness","difficulty":"beginner"},
        {"content_id":"c4","type":"yoga","duration_min":15,"intensity":"medium","goal_tag":"stress","difficulty":"beginner"},
    ])
    inter = pd.DataFrame([
        {"user_id":"u1","content_id":"c1","reward":1,"day_of_week":2,"hour_bucket":"morning","arm":"push_morning"},
        {"user_id":"u1","content_id":"c2","reward":0,"day_of_week":2,"hour_bucket":"morning","arm":"push_evening"},
        {"user_id":"u2","content_id":"c2","reward":1,"day_of_week":5,"hour_bucket":"evening","arm":"email_evening"},
        {"user_id":"u3","content_id":"c3","reward":1,"day_of_week":1,"hour_bucket":"morning","arm":"inapp_morning"},
    ])

    users.to_csv(data / "users.csv", index=False)
    content.to_csv(data / "content_catalog.csv", index=False)
    inter.to_csv(data / "interactions.csv", index=False)

    # persona artifacts — fit preprocessor first, then ensure KMeans
    pre = build_user_preprocessor()
    X_users = select_user_features(users)
    pre.fit(X_users)
    km = _ensure_kmeans(fit_kmeans_personas(X_users, k=3))
    save_persona(pre, km, cfg.ENCODER_PATH, cfg.PERSONA_MODEL_PATH)

    # simple LTR artifact
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from src.models.ltr import ALL

    df = inter.merge(users, on="user_id").merge(content, on="content_id")
    df["popularity"] = 0.0
    df["persona"] = "0"
    df["premium"] = df["premium"].astype(bool)
    df["push_opt_in"] = df["push_opt_in"].astype(bool)
    X = df[ALL].copy()
    y = (df["reward"] > 0).astype(int).values

    num_cols = ["age","baseline_activity_min_per_day","duration_min","day_of_week","popularity"]
    cat_cols = [c for c in ALL if c not in num_cols]

    preproc = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipe = Pipeline([("pre", preproc), ("clf", LogisticRegression(max_iter=200))]).fit(X, y)
    joblib.dump(pipe, cfg.ARTIFACTS_DIR / "ltr_model.joblib")

def test_offline_evaluate_runs_and_writes_metrics(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data", raising=False)
    monkeypatch.setattr(cfg, "ARTIFACTS_DIR", tmp_path / "artifacts", raising=False)
    monkeypatch.setattr(cfg, "PERSONA_MODEL_PATH", tmp_path / "artifacts" / "kmeans_personas.joblib", raising=False)
    monkeypatch.setattr(cfg, "ENCODER_PATH", tmp_path / "artifacts" / "preprocess_encoder.joblib", raising=False)
    monkeypatch.setattr(cfg, "BANDIT_PATH", tmp_path / "artifacts" / "bandit_lin_ts.joblib", raising=False)

    _seed_eval_env(tmp_path)

    from scripts.evaluate import evaluate, precision_at_k, average_precision_at_k

    ranked = ["a","b","c","d"]
    assert precision_at_k(ranked, "b", 3) == 1.0
    assert 0.0 <= average_precision_at_k(ranked, "c", 3) <= 1.0

    m = evaluate(top_k=3)
    assert "bandit" in m and "recommender" in m
    assert (tmp_path / "artifacts" / "metrics.json").exists()
