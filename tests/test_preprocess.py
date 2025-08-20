import numpy as np
import pandas as pd
from src.features.preprocess import build_user_preprocessor, select_user_features, NUM, CAT

def test_preprocessor_dense_output_and_missing_handling():
    df = pd.DataFrame([
        {"user_id":"u1","age":30,"gender":"female","work_pattern":"9-5",
         "primary_goal":"stress","baseline_activity_min_per_day":10,"premium":True,
         "push_opt_in":True,"chronotype":"morning","language":"en"},
        # missing & NaNs should be filled then encoded
        {"user_id":"u2","age":np.nan,"gender":None,"work_pattern":None,
         "primary_goal":None,"baseline_activity_min_per_day":np.nan,"premium":False,
         "push_opt_in":False,"chronotype":None,"language":None},
    ])
    pre = build_user_preprocessor()
    X = pre.fit_transform(select_user_features(df))
    assert X.shape[0] == 2
    # ensure no sparse matrix sneaks through and no NaNs
    assert isinstance(X, np.ndarray) and np.isfinite(X).all()
    # select_user_features type/na hygiene
    clean = select_user_features(df)
    assert clean[NUM].isna().sum().sum() == 0
    assert clean[CAT].isna().sum().sum() == 0
