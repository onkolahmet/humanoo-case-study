from __future__ import annotations
from pathlib import Path
import joblib

def save_obj(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_obj(path: Path):
    return joblib.load(path)
