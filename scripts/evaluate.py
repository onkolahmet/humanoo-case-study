from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATA_DIR, ARTIFACTS_DIR, ARMS, BANDIT_D, BANDIT_PATH, ENCODER_PATH, PERSONA_MODEL_PATH
from src.models.bandit import LinTSBandit
from src.models.ltr import LTRModel, build_candidate_features
from src.features.persona_clustering import load as load_persona, assign_personas
from src.features.preprocess import select_user_features

METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
RNG = np.random.default_rng(42)

def precision_at_k(ranked_ids, true_id, k):
    return 1.0 if true_id in ranked_ids[:k] else 0.0

def average_precision_at_k(ranked_ids, true_id, k):
    if true_id in ranked_ids[:k]:
        return 1.0 / (ranked_ids.index(true_id) + 1)
    return 0.0

def _user_vec(u: pd.Series, dow: int, bucket: str) -> np.ndarray:
    return np.array([
        1.0,
        float(u.age)/60.0,
        float(u.baseline_activity_min_per_day)/60.0,
        1.0 if bool(u.premium) else 0.0,
        1.0 if bool(u.push_opt_in) else 0.0,
        1.0 if str(u.chronotype)=="morning" else 0.0,
        1.0 if str(u.primary_goal)=="stress" else 0.0,
        1.0 if str(u.primary_goal)=="weight_loss" else 0.0,
        1.0 if bucket=="morning" else 0.0,
        float(dow)/6.0,
    ], dtype=float)

def evaluate(top_k: int = 5) -> dict:
    users = pd.read_csv(DATA_DIR / "users.csv")
    content = pd.read_csv(DATA_DIR / "content_catalog.csv")
    inter = pd.read_csv(DATA_DIR / "interactions.csv").sample(frac=1.0, random_state=42).reset_index(drop=True)

    # popularity for content
    pop = inter.groupby("content_id")["reward"].mean().rename("popularity").reset_index()
    content = content.merge(pop, on="content_id", how="left")
    content["popularity"] = content["popularity"].fillna(0.0)

    # personas
    pre, km = load_persona(ENCODER_PATH, PERSONA_MODEL_PATH)
    users_p = assign_personas(select_user_features(users), pre, km).set_index("user_id")

    # split
    n = len(inter); split = int(0.8 * n)
    train_inter = inter.iloc[:split].copy()
    test_inter  = inter.iloc[split:].copy()

    # bandit
    bandit = LinTSBandit(ARMS, d=BANDIT_D, alpha=0.5)
    users_idx = users.set_index("user_id")
    for _, r in train_inter.iterrows():
        u = users_idx.loc[r.user_id]
        x = _user_vec(u, int(r.day_of_week), str(r.hour_bucket))
        bandit.update(str(r.arm), float(r.reward), x)

    matches, match_rewards, test_rewards = [], [], []
    for _, r in test_inter.iterrows():
        u = users_idx.loc[r.user_id]
        x = _user_vec(u, int(r.day_of_week), str(r.hour_bucket))
        arm = bandit.choose(x)
        matches.append(1 if arm == r.arm else 0)
        if arm == r.arm: match_rewards.append(float(r.reward))
        test_rewards.append(float(r.reward))
    bandit_metrics = {
        "policy_match_rate": round(float(np.mean(matches)), 4) if matches else 0.0,
        "matched_ctr": round(float(np.mean(match_rewards)), 4) if match_rewards else 0.0,
        "overall_ctr_baseline": round(float(np.mean(test_rewards)), 4) if test_rewards else 0.0,
    }

    # learned scorer (LTR)
    ltr_path = ARTIFACTS_DIR / "ltr_model.joblib"
    if not ltr_path.exists():
        raise RuntimeError("artifacts/ltr_model.joblib not found. Run `make train-ltr` or `make train`.")
    ltr = LTRModel(ltr_path)

    hits, aps = [], []
    pos = test_inter[test_inter["reward"] == 1].copy()
    for _, r in pos.iterrows():
        u = users_idx.loc[r.user_id]
        persona = int(users_p.loc[r.user_id]["persona"]) if r.user_id in users_p.index else 0

        pool = content[content["goal_tag"] == str(u.primary_goal)].copy()
        if pool.empty: pool = content.copy()

        feats = build_candidate_features(pool, u, int(r.day_of_week), str(r.hour_bucket), persona)
        scores = ltr.predict_proba(feats)
        pool = pool.assign(score=scores.values)
        ranked_ids = pool.sort_values("score", ascending=False).head(max(top_k, 20))["content_id"].astype(str).tolist()

        hits.append(precision_at_k(ranked_ids, str(r.content_id), top_k))
        aps.append(average_precision_at_k(ranked_ids, str(r.content_id), top_k))

    rec_metrics = {
        "hit_rate@k": round(float(np.mean(hits)), 4) if hits else 0.0,
        "map@k": round(float(np.mean(aps)), 4) if aps else 0.0,
        "k": int(top_k),
        "positives_in_test": int(len(pos)),
        "notes": "Learned scorer: probabilities from LTR model; candidate pool goal-filtered.",
    }

    metrics = {"bandit": bandit_metrics, "recommender": rec_metrics}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics

if __name__ == "__main__":
    m = evaluate(top_k=5)
