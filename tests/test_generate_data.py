# tests/test_generate_data.py
import os
import shutil
import pandas as pd
import numpy as np
import pytest
import random
from pathlib import Path

import scripts.generate_data as gd


def test_gen_users_shape_and_columns():
    df = gd.gen_users(50)
    assert df.shape[0] == 50
    expected_cols = {
        "user_id", "age", "gender", "work_pattern", "primary_goal",
        "baseline_activity_min_per_day", "premium", "push_opt_in",
        "chronotype", "language"
    }
    assert set(df.columns) == expected_cols
    # user_id format
    assert all(df["user_id"].str.startswith("u"))


def test_gen_content_shape_and_columns():
    df = gd.gen_content(30)
    assert df.shape[0] == 30
    expected_cols = {"content_id", "type", "duration_min", "intensity", "goal_tag", "difficulty"}
    assert set(df.columns) == expected_cols
    assert all(df["content_id"].str.startswith("c"))
    assert df["duration_min"].between(8, 35).all()


def test_prop_probability_bounds_and_boosts():
    u = {
        "primary_goal": "weight_loss",
        "baseline_activity_min_per_day": 15,
        "chronotype": "morning"
    }
    c = {
        "goal_tag": "weight_loss",
        "intensity": "low",
        "duration_min": 15,
        "type": "walk",
        "difficulty": "beginner"
    }
    p = gd.prop(u, c, dow=2, bucket="morning")
    # always between [0.01, 0.95]
    assert 0.01 <= p <= 0.95
    # aligned goal should increase probability significantly
    assert p > 0.4


def test_gen_interactions_consistency():
    users = gd.gen_users(20)
    items = gd.gen_content(10)
    inter = gd.gen_interactions(users, items, n=200)
    expected_cols = {"user_id", "content_id", "arm", "reward", "day_of_week", "hour_bucket"}
    assert set(inter.columns) == expected_cols
    # user/content IDs valid
    assert set(inter["user_id"]).issubset(set(users["user_id"]))
    assert set(inter["content_id"]).issubset(set(items["content_id"]))
    # arms valid
    valid_arms = {"push_morning","push_evening","email_morning","email_evening","inapp_morning","inapp_evening"}
    assert set(inter["arm"]).issubset(valid_arms)
    # reward only 0 or 1
    assert set(inter["reward"]).issubset({0, 1})
    # day_of_week between 0 and 6
    assert inter["day_of_week"].between(0, 6).all()
    # bucket values valid
    assert set(inter["hour_bucket"]).issubset({"morning", "evening"})


def test_reproducibility_with_seed():
    # Reset RNGs before each call
    gd.rng = np.random.default_rng(42)
    random.seed(42)
    df1 = gd.gen_users(10)

    gd.rng = np.random.default_rng(42)
    random.seed(42)
    df2 = gd.gen_users(10)

    pd.testing.assert_frame_equal(df1, df2)



def test_main_creates_csvs(tmp_path):
    # override BASE temporarily
    old_base = gd.BASE
    gd.BASE = tmp_path
    try:
        gd.main()
        # check files created
        for fname in ["users.csv", "content_catalog.csv", "interactions.csv"]:
            fpath = tmp_path / fname
            assert fpath.exists()
            df = pd.read_csv(fpath)
            assert not df.empty
    finally:
        gd.BASE = old_base
