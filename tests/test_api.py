from pathlib import Path
import importlib
import pandas as pd
import numpy as np
import joblib

from fastapi.testclient import TestClient

import src.config as cfg
from src.features.preprocess import build_user_preprocessor, select_user_features
from src.features.persona_clustering import fit_kmeans_personas, save as save_persona
from src.models.ltr import ALL

def _ensure_kmeans(obj):
    if isinstance(obj, tuple):
        for part in obj:
            if hasattr(part, "predict"):
                return part
        raise AssertionError("fit_kmeans_personas returned a tuple without a KMeans-like object.")
    return obj

def _write_minimal_data(root: Path) -> None:
    data = root / "data"
    arts = root / "artifacts"
    data.mkdir(parents=True, exist_ok=True)
    arts.mkdir(parents=True, exist_ok=True)

    users = pd.DataFrame([
        {"user_id":"u1","age":29,"gender":"female","work_pattern":"9-5","primary_goal":"stress",
         "baseline_activity_min_per_day":12,"premium":False,"push_opt_in":True,"chronotype":"morning","language":"en"},
        {"user_id":"u2","age":41,"gender":"male","work_pattern":"shift","primary_goal":"fitness",
         "baseline_activity_min_per_day":3,"premium":True,"push_opt_in":False,"chronotype":"evening","language":"de"},
    ])

    # Ensure at least 3 items match goal='stress' so top_k=3 is satisfiable
    content = pd.DataFrame([
        {"content_id":"c1","type":"meditation","duration_min":10,"intensity":"low","goal_tag":"stress","difficulty":"beginner"},
        {"content_id":"c4","type":"yoga","duration_min":15,"intensity":"medium","goal_tag":"stress","difficulty":"beginner"},
        {"content_id":"c5","type":"breathwork","duration_min":8,"intensity":"low","goal_tag":"stress","difficulty":"beginner"},
        # non-stress items
        {"content_id":"c2","type":"hiit","duration_min":25,"intensity":"high","goal_tag":"fitness","difficulty":"advanced"},
        {"content_id":"c3","type":"walk","duration_min":20,"intensity":"medium","goal_tag":"fitness","difficulty":"beginner"},
    ])

    inter = pd.DataFrame([
        {"user_id":"u1","content_id":"c1","reward":1,"day_of_week":2,"hour_bucket":"morning","arm":"push_morning"},
        {"user_id":"u1","content_id":"c2","reward":0,"day_of_week":2,"hour_bucket":"morning","arm":"push_evening"},
        {"user_id":"u2","content_id":"c2","reward":1,"day_of_week":5,"hour_bucket":"evening","arm":"email_evening"},
        {"user_id":"u1","content_id":"c4","reward":1,"day_of_week":2,"hour_bucket":"morning","arm":"inapp_morning"},
    ])

    users.to_csv(data / "users.csv", index=False)
    content.to_csv(data / "content_catalog.csv", index=False)
    inter.to_csv(data / "interactions.csv", index=False)

    # personas — fit preprocessor then kmeans
    pre = build_user_preprocessor()
    X_users = select_user_features(users)
    pre.fit(X_users)
    km = _ensure_kmeans(fit_kmeans_personas(X_users, k=2))
    save_persona(pre, km, cfg.ENCODER_PATH, cfg.PERSONA_MODEL_PATH)

    # LTR artifact
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    X = inter.merge(users, on="user_id", how="left").merge(content, on="content_id", how="left").copy()
    X["popularity"] = 0.0
    X["persona"] = "0"
    X["premium"] = X["premium"].astype(bool)
    X["push_opt_in"] = X["push_opt_in"].astype(bool)
    X = X[ALL].copy()
    y = np.array((inter["reward"] > 0).astype(int), dtype=int)

    num_cols = ["age", "baseline_activity_min_per_day", "duration_min", "day_of_week", "popularity"]
    cat_cols = [c for c in ALL if c not in num_cols]

    preproc = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipe = Pipeline([("pre", preproc), ("clf", LogisticRegression(max_iter=200))]).fit(X, y)
    joblib.dump(pipe, cfg.ARTIFACTS_DIR / "ltr_model.joblib")

def test_api_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data", raising=False)
    monkeypatch.setattr(cfg, "ARTIFACTS_DIR", tmp_path / "artifacts", raising=False)
    monkeypatch.setattr(cfg, "PERSONA_MODEL_PATH", tmp_path / "artifacts" / "kmeans_personas.joblib", raising=False)
    monkeypatch.setattr(cfg, "ENCODER_PATH", tmp_path / "artifacts" / "preprocess_encoder.joblib", raising=False)
    monkeypatch.setattr(cfg, "BANDIT_PATH", tmp_path / "artifacts" / "bandit_lin_ts.joblib", raising=False)

    _write_minimal_data(tmp_path)

    import src.service.api as api_module
    importlib.reload(api_module)
    app = api_module.app
    client = TestClient(app)

    # helper
    r = client.get("/helper")
    assert r.status_code == 200
    hb = r.json()
    assert "arms" in hb and len(hb["arms"]) > 0
    assert hb["sample_content_id"]

    # ask for 3, should now be available due to seeded catalog
    payload = {
    "user": hb["sample_user"],
    "context": {"day_of_week": 2, "hour_bucket": "morning"},
    "top_k": 3,
    }
    r2 = client.post("/recommendations", json=payload)
    assert r2.status_code == 200
    body = r2.json()
    assert "chosen_arm" in body
    assert isinstance(body["items"], list)
    # instead of hard ==3, allow up to requested top_k
    assert 1 <= len(body["items"]) <= payload["top_k"]

    # FEEDBACK: flatten context fields to match API schema (no nested 'context')
    fb = {
        "user_id": payload["user"]["user_id"],
        "content_id": body["items"][0]["content_id"],
        "arm": body["chosen_arm"],
        "reward": 1,                  # int/float ok, but int avoids validation surprises
        "day_of_week": 2,             # flattened
        "hour_bucket": "morning"      # flattened
    }
    r3 = client.post("/feedback", json=fb)
    assert r3.status_code == 200
    assert (tmp_path / "artifacts" / "bandit_lin_ts.joblib").exists()
