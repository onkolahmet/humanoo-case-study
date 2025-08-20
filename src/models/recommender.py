from __future__ import annotations
import pandas as pd

# Persona preferences are lightweight heuristics:
PERSONA_PREFERENCES = {
    0: {"types": ["yoga", "stretch", "meditation"], "intensity": ["low", "medium"], "duration": (5, 20)},
    1: {"types": ["hiit", "cardio", "strength"],     "intensity": ["medium", "high"], "duration": (10, 30)},
    2: {"types": ["meditation", "breathwork"],       "intensity": ["low"],            "duration": (5, 15)},
    3: {"types": ["walk", "cardio", "yoga"],         "intensity": ["low", "medium"],  "duration": (10, 25)},
}
INTENSITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def score_content(row: pd.Series, user_goal: str, persona: int) -> float:
    """
    Cold-start–friendly scoring:
    - strong weight on matching the user's goal
    - persona-preferred types/intensity/duration
    - gentle penalty for intensity
    - beginner-friendly boost
    - popularity prior (CTR in [0,1]) if available on the row
    """
    prefs = PERSONA_PREFERENCES.get(int(persona), PERSONA_PREFERENCES[0])
    score = 0.0

    # Stronger prior on goal match
    if str(row["goal_tag"]) == str(user_goal):
        score += 3.0

    # Persona alignment
    if row["type"] in prefs["types"]:
        score += 1.0
    if row["intensity"] in prefs["intensity"]:
        score += 0.5
    lo, hi = prefs["duration"]
    if lo <= row["duration_min"] <= hi:
        score += 0.5

    # Nudge away from too-intense in early journey
    score -= 0.1 * INTENSITY_ORDER.get(str(row["intensity"]), 1)

    # Prefer beginner/all difficulty
    if row["difficulty"] in ("beginner", "all"):
        score += 1.0

    # Popularity prior (CTR); column may be absent on pure catalog
    pop = 0.0
    if "popularity" in row.index:
        try:
            pop = float(row["popularity"])
        except Exception:
            pop = 0.0
    score += 0.5 * pop

    return float(score)


def rank_content(df_content: pd.DataFrame, user_goal: str, persona: int, top_k: int = 5) -> pd.DataFrame:
    """
    Filter the catalog to the user's goal (if present), then score and return Top-K.
    If no items match the goal, fall back to the whole catalog.
    """
    if "goal_tag" in df_content.columns:
        df = df_content[df_content["goal_tag"] == user_goal].copy()
        if df.empty:
            df = df_content.copy()
    else:
        df = df_content.copy()

    df["score"] = df.apply(lambda r: score_content(r, user_goal, persona), axis=1)
    return df.sort_values("score", ascending=False).head(top_k)
