from __future__ import annotations
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from ..config import (
    DATA_DIR,
    ARTIFACTS_DIR,
    ENCODER_PATH,
    PERSONA_MODEL_PATH,
    BANDIT_PATH,
    ARMS,
    BANDIT_D,
)
from ..features.persona_clustering import load as load_persona_model, assign_personas
from ..features.preprocess import select_user_features
from ..models.bandit import LinTSBandit
from ..models.ltr import LTRModel, build_candidate_features
from .schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
    Feedback,
    HelperBundle,
    UserProfile,
    RequestContext,
)

app = FastAPI(title="Humanoo Retention Personalization (ML)")

# Lazy singletons
_persona = None          # tuple(preprocessor, kmeans)
_bandit: LinTSBandit | None = None
_content: pd.DataFrame | None = None
_ltr: LTRModel | None = None


def _ensure_loaded():
    """Load persona encoder/kmeans, bandit, content (with popularity), and LTR model once."""
    global _persona, _bandit, _content, _ltr

    if _persona is None:
        pre, km = load_persona_model(ENCODER_PATH, PERSONA_MODEL_PATH)
        _persona = (pre, km)

    if _bandit is None:
        if Path(BANDIT_PATH).exists():
            _bandit = LinTSBandit.load(BANDIT_PATH)
        else:
            _bandit = LinTSBandit(ARMS, d=BANDIT_D)

    if _content is None:
        _content = pd.read_csv(DATA_DIR / "content_catalog.csv")
        # Popularity prior (CTR per content_id) from interactions if available
        inter_path = DATA_DIR / "interactions.csv"
        if inter_path.exists():
            inter = pd.read_csv(inter_path)
            pop = (
                inter.groupby("content_id")["reward"]
                .mean()
                .rename("popularity")
                .reset_index()
            )
            _content = _content.merge(pop, on="content_id", how="left")
            _content["popularity"] = _content["popularity"].fillna(0.0)
        else:
            _content["popularity"] = 0.0

    if _ltr is None:
        maybe = ARTIFACTS_DIR / "ltr_model.joblib"
        if maybe.exists():
            _ltr = LTRModel(maybe)
        else:
            raise RuntimeError("Learned scorer not found. Run `make train-ltr` or `make train`.")


def _user_vector_10(user_df: pd.DataFrame, day_of_week: int, hour_bucket: str) -> np.ndarray:
    """10-D bandit features: bias, age/60, baseline/60, premium, push_opt_in, chrono_morning,
       goal_is_stress, goal_is_weight_loss, bucket_morning, day_of_week/6"""
    u = user_df.iloc[0]
    return np.array(
        [
            1.0,
            float(u.age) / 60.0,
            float(u.baseline_activity_min_per_day) / 60.0,
            1.0 if u.premium else 0.0,
            1.0 if u.push_opt_in else 0.0,
            1.0 if u.chronotype == "morning" else 0.0,
            1.0 if u.primary_goal == "stress" else 0.0,
            1.0 if u.primary_goal == "weight_loss" else 0.0,
            1.0 if hour_bucket == "morning" else 0.0,
            float(day_of_week) / 6.0,
        ],
        dtype=float,
    )

# --------- enum normalization helpers for /helper ---------
_ALLOWED_WORK = {"9-5", "shift", "flex"}
_ALLOWED_GENDER = {"male", "female", "other"}
_ALLOWED_GOAL = {"weight_loss", "stress", "fitness"}
_ALLOWED_CHRONO = {"morning", "evening"}
_ALLOWED_LANG = {"en", "de", "fr"}

def _norm_work_pattern(v: str) -> str:
    v = str(v).strip().lower()
    if v in _ALLOWED_WORK: return "9-5" if v == "9-5" else v
    return {"freelance":"flex","remote":"flex","contract":"flex","office":"9-5","nine-to-five":"9-5"}.get(v,"flex")

def _norm_gender(v: str) -> str:
    v = str(v).strip().lower()
    return v if v in _ALLOWED_GENDER else "other"

def _norm_goal(v: str) -> str:
    v = str(v).strip().lower()
    return v if v in _ALLOWED_GOAL else "fitness"

def _norm_chronotype(v: str) -> str:
    v = str(v).strip().lower()
    return v if v in _ALLOWED_CHRONO else "morning"

def _norm_lang(v: str) -> str:
    v = str(v).strip().lower()
    return v if v in _ALLOWED_LANG else "en"


# ---------- Consolidated helper (single GET) ----------

@app.get("/helper", response_model=HelperBundle)
def helper_bundle():
    """
    Single helper endpoint that returns:
    - allowed arms
    - a valid sample user (normalized)
    - a valid sample content_id
    - ready-to-send example payloads for /recommendations and /feedback
    """
    _ensure_loaded()
    assert _content is not None

    users_path = DATA_DIR / "users.csv"
    if not users_path.exists():
        raise HTTPException(status_code=404, detail="users.csv not found; run `make data`.")
    users_df = pd.read_csv(users_path)
    if users_df.empty:
        raise HTTPException(status_code=404, detail="users.csv is empty.")
    urow = users_df.sample(n=1, random_state=random.randint(0, 9999)).iloc[0]

    work_pattern = _norm_work_pattern(urow.work_pattern)
    gender = _norm_gender(urow.gender)
    primary_goal = _norm_goal(urow.primary_goal)
    chronotype = _norm_chronotype(urow.chronotype)
    language = _norm_lang(getattr(urow, "language", "en"))
    age = int(max(13, min(100, int(urow.age))))
    baseline = int(max(0, min(300, int(urow.baseline_activity_min_per_day))))

    sample_user = UserProfile(
        user_id=str(urow.user_id),
        age=age,
        gender=gender,
        work_pattern=work_pattern,
        primary_goal=primary_goal,
        baseline_activity_min_per_day=baseline,
        premium=bool(urow.premium),
        push_opt_in=bool(urow.push_opt_in),
        chronotype=chronotype,
        language=language,
    )

    dfc = _content.copy()
    if "popularity" in dfc.columns:
        dfc = dfc.sort_values("popularity", ascending=False)
    cid = str(dfc.iloc[0].content_id)

    ctx = RequestContext(
        day_of_week=random.randint(0, 6),
        hour_bucket=random.choice(["morning", "evening"]),
    )

    rec_req = RecommendationRequest(user=sample_user, context=ctx, top_k=5)
    fb = Feedback(
        user_id=sample_user.user_id,
        content_id=cid,
        arm=random.choice(ARMS),
        reward=1 if random.random() < 0.7 else 0,
        day_of_week=ctx.day_of_week,
        hour_bucket=ctx.hour_bucket,
    )

    return HelperBundle(
        arms=ARMS,
        sample_user=sample_user,
        sample_content_id=cid,
        recommend_example=rec_req,
        feedback_example=fb,
    )


# ---------- Core endpoints (learned scorer) ----------

@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):
    _ensure_loaded()
    assert _persona is not None and _bandit is not None and _content is not None and _ltr is not None

    user_df = pd.DataFrame([req.user.model_dump()])
    pre, km = _persona

    # Persona assignment
    personas_df = assign_personas(select_user_features(user_df), pre, km)
    persona = int(personas_df.iloc[0]["persona"])

    # Candidate pool (goal-filter; if empty, fall back to all)
    pool = _content[_content["goal_tag"] == req.user.primary_goal].copy()
    if pool.empty:
        pool = _content.copy()

    # Build features and score with learned model
    feats = build_candidate_features(pool, user_df.iloc[0], req.context.day_of_week,
                                     req.context.hour_bucket, persona)
    scores = _ltr.predict_proba(feats)
    pool = pool.assign(score=scores.values)
    ranked = pool.sort_values("score", ascending=False).head(req.top_k)

    # Bandit arm selection
    x = _user_vector_10(user_df, req.context.day_of_week, req.context.hour_bucket)
    if _bandit.d != len(x):
        x = np.pad(x, (0, _bandit.d - len(x))) if len(x) < _bandit.d else x[: _bandit.d]
    chosen = _bandit.choose(x)

    items = [
        RecommendationItem(
            content_id=str(r.content_id),
            type=str(r.type),
            duration_min=int(r.duration_min),
            intensity=str(r.intensity),
            goal_tag=str(r.goal_tag),
            difficulty=str(r.difficulty),
            score=float(r.score),
        )
        for _, r in ranked.iterrows()
    ]

    rationale = (
        f"Persona {persona} + goal '{req.user.primary_goal}' suggest these; "
        f"learned scorer ranked by P(reward); bandit selected '{chosen}'."
    )
    return RecommendationResponse(
        persona=persona, chosen_arm=chosen, items=items, rationale=rationale
    )


@app.post("/feedback")
def feedback(fb: Feedback):
    _ensure_loaded()
    assert _bandit is not None

    if fb.arm not in ARMS:
        raise HTTPException(status_code=400, detail=f"Invalid arm '{fb.arm}'. Allowed: {ARMS}")

    users_csv = pd.read_csv(DATA_DIR / "users.csv")
    urow = users_csv[users_csv["user_id"] == fb.user_id]
    if urow.empty:
        raise HTTPException(status_code=404, detail="user_id not found")

    x = _user_vector_10(urow, fb.day_of_week, fb.hour_bucket)
    if _bandit.d != len(x):
        x = np.pad(x, (0, _bandit.d - len(x))) if len(x) < _bandit.d else x[: _bandit.d]
    _bandit.update(fb.arm, float(fb.reward), x)
    _bandit.save(BANDIT_PATH)
    return {"status": "ok", "updated_arm": fb.arm}


@app.get("/metrics")
def get_metrics():
    """Return last saved offline evaluation metrics (written by scripts/evaluate.py)."""
    path = ARTIFACTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics not found; run offline evaluation first (e.g., `make eval`).",
        )
    return json.loads(path.read_text(encoding="utf-8"))
