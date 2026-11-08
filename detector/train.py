"""Entraîne sur train, choisit les seuils sur validation, évalue sur test."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import DATA_DIR, HF_DATASET, HF_SUBSET, META_PATH, MODEL_PATH
from .model import build_model


def load_partition(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} introuvable. Lance python -m detector.download_data")
    df = pd.read_csv(path)
    needed = {"text", "label"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Colonnes requises dans {path}: {sorted(needed)}")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df = df[(df["text"].str.len() >= 50) & (df["label"].isin([0, 1]))]
    df = df.drop_duplicates(subset=["text"], keep="first")
    if df["label"].nunique() != 2 or df["label"].value_counts().min() < 5:
        raise ValueError(f"Classes 0 et 1 obligatoires, >= 5 exemples chacune : {path}")
    return df.reset_index(drop=True)


def choose_thresholds(y_true, probabilities):
    """Seuils conservateurs sélectionnés sur validation (pas sur test).

    Seuil IA: au-delà du 95e percentile des scores de textes humains.
    Seuil humain: en deçà du 5e percentile des scores IA.
    Ces seuils sont expérimentaux, pas des garanties hors distribution.
    """
    y = np.asarray(y_true)
    probs = np.asarray(probabilities)
    human = probs[y == 0]
    artificial = probs[y == 1]
    if len(human) < 5 or len(artificial) < 5:
        raise ValueError("Validation: >= 5 échantillons par classe sont nécessaires")
    ai_threshold = float(max(0.5, np.nextafter(np.quantile(human, 0.95), 1.0)))
    human_threshold = float(min(0.5, np.nextafter(np.quantile(artificial, 0.05), 0.0)))
    human_threshold = float(max(0.0, min(human_threshold, ai_threshold - 0.01)))
    return {"human": round(human_threshold, 6), "ai": round(ai_threshold, 6)}


def evaluate(y_true, prob, threshold):
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision_ia": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "rappel_ia": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_ia": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "matrice_confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "faux_positifs_humains": int(((np.asarray(y_true) == 0) & (y_pred == 1)).sum()),
        "nombre_test": int(len(y_true)),
    }


def train(data_dir=DATA_DIR, model_path=MODEL_PATH, meta_path=META_PATH):
    data_dir = Path(data_dir)
    model_path = Path(model_path)
    meta_path = Path(meta_path)
    frames = {split: load_partition(data_dir / f"{split}.csv")
              for split in ("train", "validation", "test")}

    # Les textes identiques ne doivent jamais figurer dans plusieurs partitions.
    # On rejette les CSV suspects plutôt que de mesurer des scores trompeurs.
    split_sets = {k: set(v["text"].str.casefold()) for k, v in frames.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_sets[a] & split_sets[b]:
            raise ValueError(f"Fuite de données : doublons exacts entre {a} et {b}")

    model = build_model()
    tr, va, te = [frames[k] for k in ("train", "validation", "test")]
    print(f"Entraînement : {len(tr)} textes ; validation : {len(va)} ; test : {len(te)}")
    model.fit(tr["text"].tolist(), tr["label"].tolist())

    if list(model.named_steps["classification"].classes_) != [0, 1]:
        raise RuntimeError("Ordre des classes inattendu")
    val_scores = model.predict_proba(va["text"].tolist())[:, 1]
    thresholds = choose_thresholds(va["label"], val_scores)
    test_scores = model.predict_proba(te["text"].tolist())[:, 1]
    metrics = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": f"{HF_DATASET}/{HF_SUBSET}",
        "method": "TF-IDF mots (1-2) + caractères (3-5) + régression logistique",
        "labels": {"0": "humain", "1": "IA"},
        "size": {k: len(v) for k, v in frames.items()},
        "thresholds": thresholds,
        "test_seuil_0_5": evaluate(te["label"], test_scores, 0.5),
        "test_seuil_prudent_ia": evaluate(te["label"], test_scores, thresholds["ai"]),
        "limitations": [
            "Corpus HC3 français majoritairement traduit depuis l'anglais",
            "Générations anciennes; généralisation aux nouveaux LLM non garantie",
            "Le score est une sortie de modèle, pas une preuve d'auteur",
            "Les textes très courts et hors domaine sont particulièrement incertains",
        ],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["test_seuil_prudent_ia"], indent=2, ensure_ascii=False))
    print(f"Modèle: {model_path} | métriques: {meta_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Entraîne le détecteur local et évalue sa qualité.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--meta-path", type=Path, default=META_PATH)
    args = parser.parse_args()
    train(args.data_dir, args.model_path, args.meta_path)


if __name__ == "__main__":
    main()
