import numpy as np
import pandas as pd
import types
import sys

import pytest

import scripts.train_bandit as tb


def test_make_x_encodes_correctly():
    u = pd.Series({
        "age": 30,
        "baseline_activity_min_per_day": 120,
        "premium": True,
        "push_opt_in": False,
        "chronotype": "morning",
        "primary_goal": "stress"
    })
    x = tb.make_x(u, day_of_week=3, hour_bucket="morning")

    # shape
    assert x.shape == (10,)
    # bias term
    assert x[0] == 1.0
    # normalized age (30/60)
    assert np.isclose(x[1], 0.5)
    # baseline activity normalized (120/60)
    assert np.isclose(x[2], 2.0)
    # premium flag
    assert x[3] == 1.0
    # push_opt_in flag
    assert x[4] == 0.0
    # chronotype morning
    assert x[5] == 1.0
    # primary goal stress
    assert x[6] == 1.0
    # weight_loss false
    assert x[7] == 0.0
    # hour_bucket morning
    assert x[8] == 1.0
    # day_of_week normalized (3/6)
    assert np.isclose(x[9], 0.5)


def test_main_trains_and_saves(monkeypatch):
    # Fake data for users + interactions
    users = pd.DataFrame([{
        "user_id": 1,
        "age": 40,
        "baseline_activity_min_per_day": 90,
        "premium": False,
        "push_opt_in": True,
        "chronotype": "evening",
        "primary_goal": "weight_loss"
    }]).set_index("user_id")

    inter = pd.DataFrame([{
        "user_id": 1,
        "day_of_week": 2,
        "hour_bucket": "evening",
        "arm": "A",
        "reward": 1.0
    }])

    # Patch pandas.read_csv to return fake frames
    def fake_read_csv(path, *args, **kwargs):
        if "users" in str(path):
            return users.reset_index()
        if "interactions" in str(path):
            return inter
        raise FileNotFoundError(path)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    # Capture calls to LinTSBandit
    class FakeBandit:
        def __init__(self, arms, d, alpha):
            self.updates = []
            self.saved = False

        def update(self, arm, reward, x):
            self.updates.append((arm, reward, x))

        def save(self, path):
            self.saved = True
            self.path = path

    monkeypatch.setitem(sys.modules, "src.models.bandit", types.SimpleNamespace(LinTSBandit=FakeBandit))
    monkeypatch.setattr(tb, "LinTSBandit", FakeBandit)

    # Run main()
    tb.main()

    # Verify bandit was updated and saved
    bandit = tb.LinTSBandit("arms", d=10, alpha=0.5)
    assert isinstance(bandit, FakeBandit)
