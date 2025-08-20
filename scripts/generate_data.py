# scripts/generate_data.py
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
random.seed(42)

BASE = Path("data")
BASE.mkdir(parents=True, exist_ok=True)

N_USERS = 600
N_CONTENT = 300
N_INTERACTIONS = 9000

GENDERS = ["male", "female", "other"]
WORK = ["9-5", "shift", "flex"]
GOALS = ["weight_loss", "fitness", "stress"]
CHRONO = ["morning", "evening"]
LANGS = ["en", "de", "fr"]
TYPES = ["yoga", "walk", "strength", "meditation", "hiit"]
INTENS = ["low", "medium", "high"]
DIFF = ["beginner", "intermediate", "all"]

def gen_users(n=N_USERS):
    users = []
    for i in range(n):
        users.append(
            dict(
                user_id=f"u{i:04d}",
                age=int(rng.integers(18, 65)),
                gender=random.choice(GENDERS),
                work_pattern=random.choice(WORK),
                primary_goal=random.choices(GOALS, weights=[0.45, 0.35, 0.20])[0],
                baseline_activity_min_per_day=int(rng.integers(5, 50)),
                premium=bool(rng.integers(0, 2)),
                push_opt_in=True if rng.random() < 0.8 else False,
                chronotype=random.choices(CHRONO, weights=[0.6, 0.4])[0],
                language=random.choices(LANGS, weights=[0.7, 0.2, 0.1])[0],
            )
        )
    return pd.DataFrame(users)

def gen_content(n=N_CONTENT):
    items = []
    for j in range(n):
        goal = random.choices(GOALS, weights=[0.45, 0.35, 0.20])[0]
        ctype = random.choice(TYPES)
        intensity = random.choice(INTENS)
        duration = int(rng.integers(8, 35))
        difficulty = random.choice(DIFF)
        items.append(
            dict(
                content_id=f"c{j:04d}",
                type=ctype,
                duration_min=duration,
                intensity=intensity,
                goal_tag=goal,
                difficulty=difficulty,
            )
        )
    return pd.DataFrame(items)

def prop(u, c, dow, bucket):
    p = 0.08
    # goal alignment (primary positive signal)
    if u["primary_goal"] == c["goal_tag"]:
        p += 0.40
    # simple tailored boosts per goal
    if u["primary_goal"] == "weight_loss":
        if c["intensity"] == "low" and c["duration_min"] <= 20: p += 0.15
        if c["type"] in ["walk", "yoga"]: p += 0.07
    if u["primary_goal"] == "fitness":
        if c["intensity"] == "medium" and 15 <= c["duration_min"] <= 30: p += 0.12
        if c["type"] in ["hiit", "strength"]: p += 0.07
    if u["primary_goal"] == "stress":
        if c["type"] in ["yoga", "meditation"] and c["intensity"] == "low": p += 0.18

    # light preferences
    if c["difficulty"] in ["beginner", "all"]: p += 0.03
    if (u["chronotype"] == "morning" and bucket == "morning") or (u["chronotype"] == "evening" and bucket == "evening"):
        p += 0.03

    # duration closeness
    p += max(0, 0.08 - abs(c["duration_min"] - u["baseline_activity_min_per_day"]) / 200.0)

    return float(np.clip(p, 0.01, 0.95))

def gen_interactions(users: pd.DataFrame, items: pd.DataFrame, n=N_INTERACTIONS):
    rows = []
    ARMS = ["push_morning","push_evening","email_morning","email_evening","inapp_morning","inapp_evening"]
    for _ in range(n):
        u = users.sample(1, random_state=rng.integers(0, 1e9)).iloc[0].to_dict()
        # pick within-goal 70% of time → clearer pattern for eval
        if rng.random() < 0.7:
            pool = items[items["goal_tag"] == u["primary_goal"]]
            c = pool.sample(1, random_state=rng.integers(0, 1e9)).iloc[0].to_dict() if not pool.empty else \
                items.sample(1, random_state=rng.integers(0, 1e9)).iloc[0].to_dict()
        else:
            c = items.sample(1, random_state=rng.integers(0, 1e9)).iloc[0].to_dict()

        dow = int(rng.integers(0, 7))
        bucket = "morning" if rng.random() < 0.55 else "evening"
        p = prop(u, c, dow, bucket)
        p = float(np.clip(rng.normal(p, 0.04), 0.01, 0.99))
        reward = 1 if rng.random() < p else 0

        rows.append(
            dict(
                user_id=u["user_id"],
                content_id=c["content_id"],
                arm=random.choice(ARMS),
                reward=reward,
                day_of_week=dow,
                hour_bucket=bucket,
            )
        )
    return pd.DataFrame(rows)

def main():
    users = gen_users()
    content = gen_content()
    inter = gen_interactions(users, content)

    users.to_csv(BASE / "users.csv", index=False)
    content.to_csv(BASE / "content_catalog.csv", index=False)
    inter.to_csv(BASE / "interactions.csv", index=False)
    print(f"Mock data written to {BASE}")

if __name__ == "__main__":
    main()
