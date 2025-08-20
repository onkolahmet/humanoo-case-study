from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

NUM = ["age","baseline_activity_min_per_day"]
CAT = ["gender","work_pattern","primary_goal","premium","push_opt_in","chronotype","language"]

def _onehot_dense():
    # sklearn <1.2: OneHotEncoder(..., sparse=False)
    # sklearn >=1.2: OneHotEncoder(..., sparse_output=False)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_user_preprocessor() -> Pipeline:
    numeric = Pipeline(steps=[("scaler", StandardScaler())])
    categorical = Pipeline(steps=[("onehot", _onehot_dense())])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUM),
            ("cat", categorical, CAT),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # force dense output
    )

def select_user_features(df_users: pd.DataFrame) -> pd.DataFrame:
    out = df_users.copy()
    out["premium"] = out["premium"].astype(bool)
    out["push_opt_in"] = out["push_opt_in"].astype(bool)
    out[NUM] = out[NUM].fillna(0)
    out[CAT] = out[CAT].fillna("unknown")
    return out
