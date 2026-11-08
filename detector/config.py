"""Constantes partagées."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "detecteur_fr.joblib"
META_PATH = MODEL_DIR / "metrics.json"
CACHE_DIR = DATA_DIR / "cache"

HF_DATASET = "almanach/hc3_french_ood"
HF_ARCHIVE_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/data.tar.gz"
RANDOM_SEED = 42
MAX_TEXT_CHARS = 50_000

# Petit modèle de langue local (étape « perplexité », facultative).
# Observateur = modèle de base, exécutant = version instruite, même tokenizer
# (méthode Binoculars, Hans et al. 2024).
LM_OBSERVER = "Qwen/Qwen2.5-0.5B"
LM_PERFORMER = "Qwen/Qwen2.5-0.5B-Instruct"
LM_MAX_TOKENS = 256
