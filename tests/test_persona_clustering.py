from pathlib import Path
import pandas as pd
from src.features.persona_clustering import fit_kmeans_personas, assign_personas, save, load
from src.features.preprocess import build_user_preprocessor, select_user_features

def _ensure_kmeans(obj):
    # Support both: returns KMeans OR (pre, kmeans)
    if isinstance(obj, tuple):
        for part in obj:
            if hasattr(part, "predict"):
                return part
        raise AssertionError("fit_kmeans_personas returned a tuple without a KMeans-like object.")
    return obj

def test_persona_fit_assign_save_load(tmp_path: Path):
    users = pd.DataFrame([
        {"user_id":"u1","age":25,"gender":"female","work_pattern":"9-5","primary_goal":"fitness",
         "baseline_activity_min_per_day":15,"premium":True,"push_opt_in":True,"chronotype":"morning","language":"en"},
        {"user_id":"u2","age":48,"gender":"male","work_pattern":"shift","primary_goal":"stress",
         "baseline_activity_min_per_day":5,"premium":False,"push_opt_in":False,"chronotype":"evening","language":"de"},
        {"user_id":"u3","age":35,"gender":"other","work_pattern":"flex","primary_goal":"weight_loss",
         "baseline_activity_min_per_day":30,"premium":True,"push_opt_in":False,"chronotype":"morning","language":"en"},
    ])

    pre = build_user_preprocessor()
    X_users = select_user_features(users)
    pre.fit(X_users)

    km_raw = fit_kmeans_personas(X_users, k=2)
    km = _ensure_kmeans(km_raw)

    out = assign_personas(users, pre, km)
    assert "persona" in out.columns and set(out["persona"].unique()) <= {0, 1}

    enc_p = tmp_path / "enc.joblib"
    km_p  = tmp_path / "km.joblib"
    save(pre, km, enc_p, km_p)

    pre2, km2 = load(enc_p, km_p)
    out2 = assign_personas(users, pre2, km2)
    assert out2["persona"].equals(out["persona"])
