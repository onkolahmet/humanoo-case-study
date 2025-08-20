from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.models.ltr import build_candidate_features, LTRModel, ALL
from src.features.preprocess import select_user_features

def _dummy_ltr_artifact(path: Path):
    # very small pipeline with the exact columns the model will see
    num = ["age","baseline_activity_min_per_day","duration_min","day_of_week","popularity"]
    cat = ["premium","push_opt_in","chronotype","primary_goal","type","intensity",
           "difficulty","goal_tag","hour_bucket","persona"]
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=200))])

    # fabricate training rows
    X = pd.DataFrame([
        # positive
        {"age":30,"baseline_activity_min_per_day":20,"duration_min":10,"day_of_week":2,"popularity":0.2,
         "premium":True,"push_opt_in":True,"chronotype":"morning","primary_goal":"stress",
         "type":"meditation","intensity":"low","difficulty":"beginner","goal_tag":"stress",
         "hour_bucket":"morning","persona":"2"},
        # negative
        {"age":45,"baseline_activity_min_per_day":5,"duration_min":30,"day_of_week":5,"popularity":0.0,
         "premium":False,"push_opt_in":False,"chronotype":"evening","primary_goal":"fitness",
         "type":"hiit","intensity":"high","difficulty":"advanced","goal_tag":"fitness",
         "hour_bucket":"evening","persona":"1"},
    ])
    y = np.array([1,0], dtype=int)
    pipe.fit(X, y)
    joblib.dump(pipe, path)

def test_build_features_and_predict(tmp_path: Path):
    # candidate pool
    content = pd.DataFrame([
        {"content_id":"c1","type":"meditation","duration_min":10,"intensity":"low","goal_tag":"stress","difficulty":"beginner","popularity":0.1},
        {"content_id":"c2","type":"hiit","duration_min":30,"intensity":"high","goal_tag":"fitness","difficulty":"advanced","popularity":0.0},
    ])
    # user row as a Series (matches service)
    user_row = pd.Series({
        "user_id":"u1","age":30,"gender":"female","work_pattern":"9-5",
        "primary_goal":"stress","baseline_activity_min_per_day":20,"premium":True,
        "push_opt_in":True,"chronotype":"morning","language":"en"
    })
    feats = build_candidate_features(content, user_row, day_of_week=2, hour_bucket="morning", persona=2)
    assert list(feats.columns) == ALL and len(feats) == 2

    # trained artifact + wrapper
    model_path = tmp_path / "ltr.joblib"
    _dummy_ltr_artifact(model_path)
    model = LTRModel(model_path)
    proba = model.predict_proba(feats)
    assert proba.shape == (2,)

    # should rank meditation/stress higher given training signal
    assert float(proba.iloc[0]) > float(proba.iloc[1])
