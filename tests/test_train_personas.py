import pandas as pd
import types
import sys

import scripts.train_personas as tp


def test_main_calls_fit_and_save(monkeypatch):
    # Fake dataframe for users
    fake_users = pd.DataFrame([
        {"user_id": 1, "age": 25, "baseline_activity_min_per_day": 60},
        {"user_id": 2, "age": 35, "baseline_activity_min_per_day": 120},
    ])

    # Patch pandas.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda path: fake_users)

    # Capture calls
    called = {}

    def fake_fit_kmeans_personas(df, k):
        called["fit_df"] = df
        called["fit_k"] = k
        return "preproc", "kmeans"

    def fake_save(pre, km, ep, pp):
        called["saved"] = (pre, km, ep, pp)

    monkeypatch.setattr(tp, "fit_kmeans_personas", fake_fit_kmeans_personas)
    monkeypatch.setattr(tp, "save", fake_save)

    # Run main()
    tp.main()

    # Assertions
    assert "fit_df" in called
    assert called["fit_df"].equals(fake_users)
    assert called["fit_k"] == 4
    assert called["saved"][0] == "preproc"
    assert called["saved"][1] == "kmeans"
    assert called["saved"][2] == tp.ENCODER_PATH
    assert called["saved"][3] == tp.PERSONA_MODEL_PATH
