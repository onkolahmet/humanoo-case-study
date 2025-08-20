from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

# Keep feature names centralized (must match training script)
NUM = ["age", "baseline_activity_min_per_day", "duration_min", "day_of_week", "popularity"]
CAT = ["premium", "push_opt_in", "chronotype", "primary_goal", "type",
       "intensity", "difficulty", "goal_tag", "hour_bucket", "persona"]
ALL = NUM + CAT

class LTRModel:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.pipe = joblib.load(self.path)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        # expects ALL columns present
        probs = self.pipe.predict_proba(X[ALL])[:, 1]
        return pd.Series(probs, index=X.index, name="score")

def build_candidate_features(cands: pd.DataFrame,
                             user_row: pd.Series,
                             day_of_week: int,
                             hour_bucket: str,
                             persona: int) -> pd.DataFrame:
    """
    Enrich candidate content rows with user+context features expected by the model.
    Returns a DataFrame with ALL columns in the right dtypes.
    """
    df = cands.copy()
    # numerical
    df["age"] = int(user_row["age"])
    df["baseline_activity_min_per_day"] = int(user_row["baseline_activity_min_per_day"])
    df["day_of_week"] = int(day_of_week)
    # ensure popularity exists
    if "popularity" not in df.columns:
        df["popularity"] = 0.0

    # categorical
    df["premium"] = bool(user_row["premium"])
    df["push_opt_in"] = bool(user_row["push_opt_in"])
    df["chronotype"] = str(user_row["chronotype"])
    df["primary_goal"] = str(user_row["primary_goal"])
    df["hour_bucket"] = str(hour_bucket)
    df["persona"] = str(persona)  # categorical

    # order + fill
    for col in ALL:
        if col not in df.columns:
            # duration_min, difficulty, type, intensity, goal_tag come from content df
            df[col] = 0 if col in NUM else "unknown"
    return df[ALL]
