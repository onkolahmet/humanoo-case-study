from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.config import DATA_DIR, ENCODER_PATH, PERSONA_MODEL_PATH
from src.features.persona_clustering import fit_kmeans_personas, save

def main():
    dfu = pd.read_csv(DATA_DIR / "users.csv")
    pre, km = fit_kmeans_personas(dfu, k=4)
    save(pre, km, ENCODER_PATH, PERSONA_MODEL_PATH)
    print("Saved personas to artifacts.")

if __name__ == "__main__":
    main()
