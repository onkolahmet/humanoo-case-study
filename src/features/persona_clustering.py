from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .preprocess import build_user_preprocessor, select_user_features

def _ensure_finite(X):
    X = np.array(X, dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def fit_kmeans_personas(df_users: pd.DataFrame, k: int = 4):
    pre = build_user_preprocessor()
    X = pre.fit_transform(select_user_features(df_users))
    X = _ensure_finite(X)
    kmeans = KMeans(n_clusters=k, n_init=10, algorithm="lloyd", random_state=42).fit(X)
    return pre, kmeans

def assign_personas(df_users: pd.DataFrame, pre, kmeans) -> pd.DataFrame:
    X = pre.transform(select_user_features(df_users))
    X = _ensure_finite(X)
    personas = kmeans.predict(X)
    out = df_users.copy()
    out["persona"] = personas
    return out

def save(pre, kmeans, encoder_path: Path, model_path: Path) -> None:
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, encoder_path)
    joblib.dump(kmeans, model_path)

def load(encoder_path: Path, model_path: Path):
    pre = joblib.load(encoder_path)
    km = joblib.load(model_path)
    return pre, km
