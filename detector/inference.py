"""Prédiction partagée par l'interface, l'API et le terminal."""

import json
import re
from pathlib import Path

import joblib

from .config import MAX_TEXT_CHARS, META_PATH, MODEL_PATH


def load_detector(model_path=MODEL_PATH, meta_path=META_PATH):
    model_path = Path(model_path)
    meta_path = Path(meta_path)
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            "Modèle non entraîné : exécute d'abord 'python -m detector.download_data', "
            "puis 'python -m detector.train'."
        )
    # Attention : joblib / pickle ne doit jamais charger des modèles non fiables.
    return joblib.load(model_path), json.loads(meta_path.read_text(encoding="utf-8"))


def text_statistics(text):
    words = re.findall(r"\b[\wÀ-ÿ'-]+\b", text, flags=re.UNICODE)
    return {
        "mots": len(words),
        "caracteres": len(text),
        "phrases_approx": len(re.findall(r"[.!?]+", text)) or (1 if words else 0),
    }


def analyze(text, model, metadata):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Saisis un texte non vide")
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Le texte dépasse {MAX_TEXT_CHARS:,} caractères")
    stats = text_statistics(text)
    probability_ai = float(model.predict_proba([text])[0, 1])
    low = float(metadata["thresholds"]["human"])
    high = float(metadata["thresholds"]["ai"])
    if probability_ai >= high:
        category = "indices_ia"
        message = "Le modèle détecte des indices associés aux textes générés par IA."
    elif probability_ai <= low:
        category = "indices_humains"
        message = "Le modèle détecte des indices associés aux textes humains."
    else:
        category = "indetermine"
        message = "Le modèle ne distingue pas clairement l'origine du texte."
    warnings = ["Ce résultat statistique ne prouve jamais l'identité de l'auteur."]
    if stats["mots"] < 80:
        warnings.append("Texte court (moins de 80 mots) : interprétation particulièrement fragile.")
    return {
        "categorie": category,
        "message": message,
        "score_ia": round(probability_ai, 4),
        "score_humain": round(1 - probability_ai, 4),
        "statistiques": stats,
        "avertissements": warnings,
    }
