"""Constantes partagées."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "detecteur_fr.joblib"
META_PATH = MODEL_DIR / "metrics.json"

HF_DATASET = "almanach/hc3_french_ood"
HF_SUBSET = "hc3_fr_qa"  # 'answer' = texte analysé, sans la question
HF_ENDPOINT = "https://datasets-server.huggingface.co/rows"
RANDOM_SEED = 42
MAX_TEXT_CHARS = 50_000
