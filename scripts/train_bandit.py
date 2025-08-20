from __future__ import annotations
import numpy as np, pandas as pd
from src.config import DATA_DIR, BANDIT_PATH, ARMS, BANDIT_D
from src.models.bandit import LinTSBandit

def make_x(u: pd.Series, day_of_week: int, hour_bucket: str) -> np.ndarray:
    return np.array([
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
    ], dtype=float)

def main():
    users = pd.read_csv(DATA_DIR / "users.csv").set_index("user_id")
    inter = pd.read_csv(DATA_DIR / "interactions.csv")
    bandit = LinTSBandit(ARMS, d=BANDIT_D, alpha=0.5)
    for _, row in inter.iterrows():
        u = users.loc[row.user_id]
        x = make_x(u, int(row.day_of_week), str(row.hour_bucket))
        bandit.update(str(row.arm), float(row.reward), x)
    bandit.save(BANDIT_PATH)
    print("Bandit trained & saved.")

if __name__ == "__main__":
    main()
