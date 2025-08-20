# tests/test_train_ltr.py
import pandas as pd
import numpy as np
import pytest

# IMPORTANT: import the right module path used in your repo
import scripts.train_ltr as ltr


@pytest.fixture
def fake_data(monkeypatch):
    # Minimal, schema-correct fake CSVs
    users = pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "age": [25, 40, 35, 50],
        "gender": ["female", "male", "other", "female"],
        "work_pattern": ["9-5", "shift", "flex", "9-5"],
        "primary_goal": ["stress", "fitness", "weight_loss", "stress"],
        "baseline_activity_min_per_day": [30, 45, 20, 25],
        "premium": [True, False, False, True],
        "push_opt_in": [True, False, True, True],
        "chronotype": ["morning", "evening", "morning", "evening"],
        "language": ["en", "de", "en", "fr"],
    })

    content = pd.DataFrame({
        "content_id": ["c1", "c2", "c3", "c4"],
        "type": ["yoga", "hiit", "meditation", "walk"],
        "duration_min": [15, 25, 10, 20],
        "intensity": ["low", "high", "low", "medium"],
        "goal_tag": ["stress", "fitness", "stress", "fitness"],
        "difficulty": ["beginner", "advanced", "beginner", "intermediate"],
    })

    interactions = pd.DataFrame({
        "user_id": ["u1", "u2", "u1", "u3", "u4", "u2"],
        "content_id": ["c1", "c2", "c3", "c4", "c1", "c3"],
        "reward": [1, 0, 1, 0, 1, 0],
        "day_of_week": [1, 2, 3, 4, 5, 6],
        "hour_bucket": ["morning", "evening", "morning", "evening", "morning", "evening"],
    })

    def fake_read_csv(path, *args, **kwargs):
        p = str(path)
        if "users.csv" in p:
            return users.copy()
        if "content_catalog.csv" in p:
            return content.copy()
        if "interactions.csv" in p:
            return interactions.copy()
        raise AssertionError(f"Unexpected read_csv path: {p}")

    # Patch file I/O
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    # Patch persona loader + assignment (avoid real models)
    def fake_load_persona(enc_path, km_path):
        return "pre", "km"

    def fake_assign_personas(df_users, pre, km):
        # Return minimal mapping (must contain user_id, persona)
        return pd.DataFrame({
            "user_id": users["user_id"],
            "persona": ["0", "1", "0", "1"],
        })

    monkeypatch.setattr(ltr, "load_persona", fake_load_persona)
    monkeypatch.setattr(ltr, "assign_personas", fake_assign_personas)

    return users, content, interactions


def test_build_dataset(fake_data):
    X, y = ltr.build_dataset()
    # Shape alignment
    assert len(X) == len(y) > 0

    # All expected feature columns exist
    expected = set(ltr.NUM + ltr.CAT)
    assert expected.issubset(set(X.columns)), f"Missing columns: {expected - set(X.columns)}"

    # Dtype hygiene from script
    assert X["premium"].dtype == bool
    assert X["push_opt_in"].dtype == bool
    # persona should be categorical-like (string/object)
    assert X["persona"].dtype == object or X["persona"].dtype.name == "string"

    # Basic value sanity
    assert set(np.unique(y)).issubset({0, 1})


def test_choose_estimator_returns_model():
    y = np.array([0, 1, 0, 1, 1, 0])
    clf, name = ltr.choose_estimator(y)
    assert hasattr(clf, "fit"), "Estimator must implement .fit"
    assert name in ("xgboost", "logreg")


def test_train_runs(monkeypatch, fake_data, capsys):
    # Avoid writing model to disk
    saved = {}
    def fake_dump(obj, path):
        saved["ok"] = True

    monkeypatch.setattr(ltr.joblib, "dump", fake_dump)

    # Run training end-to-end on fake data
    ltr.train()

    assert saved.get("ok", False), "Model was not 'saved' via joblib.dump"

    # Check that metrics were printed
    out = capsys.readouterr().out
    assert "[ltr]" in out and "model=" in out and "ROC-AUC=" in out
