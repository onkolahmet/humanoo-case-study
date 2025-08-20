from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

PERSONA_MODEL_PATH = ARTIFACTS_DIR / "kmeans_personas.joblib"
ENCODER_PATH = ARTIFACTS_DIR / "preprocess_encoder.joblib"
BANDIT_PATH = ARTIFACTS_DIR / "bandit_lin_ts.joblib"

ARMS = [
    "push_morning", "push_evening",
    "email_morning", "email_evening",
    "inapp_morning", "inapp_evening"
]

# >>> New: dimension of bandit feature vector (incl. hour + day)
BANDIT_D = 10
