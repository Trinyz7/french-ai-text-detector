"""Entraîne sur train, calibre et choisit les seuils sur validation, évalue sur test.

Trois évaluations complémentaires sont enregistrées dans metrics.json :

* ``test`` : HC3 traduit, mêmes domaines que l'entraînement ;
* ``test_natif`` : textes rédigés directement en français (hors distribution) ;
* ``hors_domaine`` : pour chaque domaine, un modèle entraîné *sans* ce domaine
  est évalué dessus (leave-one-domain-out). C'est la mesure la plus honnête de
  la capacité à généraliser à des sujets jamais vus.
"""

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

from .config import DATA_DIR, HF_DATASET, META_PATH, MODEL_PATH
from .model import Detector, build_pipeline

# Écart minimal autour de 0,5 : sous ces bornes, on préfère dire « indéterminé ».
MIN_AI_THRESHOLD = 0.65
MAX_HUMAN_THRESHOLD = 0.35

CONFIGS = {
    "complet": dict(words=True, chars=True, style=True, llm=True),
    "sans_llm": dict(words=True, chars=True, style=True, llm=False),
    "tfidf": dict(words=True, chars=True, style=False, llm=False),
    "style_llm": dict(words=False, chars=False, style=True, llm=True),
    "style": dict(words=False, chars=False, style=True, llm=False),
}


def load_partition(path, required=True):
    path = Path(path)
    if not path.is_file():
        if not required:
            return None
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
    if "source" not in df.columns:
        df["source"] = "inconnu"
    df["source"] = df["source"].fillna("inconnu").astype(str).str.replace(r"^hc3/", "", regex=True)
    if df["label"].nunique() != 2 or df["label"].value_counts().min() < 5:
        raise ValueError(f"Classes 0 et 1 obligatoires, >= 5 exemples chacune : {path}")
    return df.reset_index(drop=True)


def choose_thresholds(y_true, probabilities):
    """Seuils conservateurs sélectionnés sur validation (pas sur test).

    Seuil IA : au-delà du 95e percentile des scores de textes humains (>= 0,65).
    Seuil humain : en deçà du 5e percentile des scores IA (<= 0,35).
    Entre les deux, le résultat est « indéterminé ».
    """
    y = np.asarray(y_true)
    probs = np.asarray(probabilities)
    human = probs[y == 0]
    artificial = probs[y == 1]
    if len(human) < 5 or len(artificial) < 5:
        raise ValueError("Validation: >= 5 échantillons par classe sont nécessaires")
    ai_threshold = float(max(MIN_AI_THRESHOLD, np.nextafter(np.quantile(human, 0.95), 1.0)))
    human_threshold = float(min(MAX_HUMAN_THRESHOLD, np.nextafter(np.quantile(artificial, 0.05), 0.0)))
    ai_threshold = min(ai_threshold, 0.99)
    human_threshold = max(0.01, min(human_threshold, ai_threshold - 0.01))
    return {"human": round(human_threshold, 6), "ai": round(ai_threshold, 6)}


def evaluate(y_true, prob, threshold):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    both = len(set(y_true)) == 2
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4) if both else None,
        "precision_ia": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "rappel_ia": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_ia": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4) if both else None,
        "matrice_confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "taux_faux_positifs": round(float(((y_true == 0) & (y_pred == 1)).sum() / max(1, (y_true == 0).sum())), 4),
        "faux_positifs_humains": int(((y_true == 0) & (y_pred == 1)).sum()),
        "nombre_test": int(len(y_true)),
    }


def decision_summary(y_true, prob, thresholds):
    """Répartition des décisions à trois issues (humain / indéterminé / IA)."""
    y = np.asarray(y_true)
    p = np.asarray(prob)
    out = {}
    for label, name in ((0, "humains"), (1, "ia")):
        sel = p[y == label]
        if len(sel) == 0:
            continue
        out[name] = {
            "classes_ia": round(float((sel >= thresholds["ai"]).mean()), 4),
            "indetermines": round(float(((sel > thresholds["human"]) & (sel < thresholds["ai"])).mean()), 4),
            "classes_humains": round(float((sel <= thresholds["human"]).mean()), 4),
        }
    return out


def per_source(frame, prob, threshold):
    rows = {}
    for source, idx in frame.groupby("source").groups.items():
        y = frame.loc[idx, "label"].to_numpy()
        p = np.asarray(prob)[np.asarray(idx)]
        pred = p >= threshold
        rows[source] = {
            "n": int(len(y)),
            "taux_faux_positifs": round(float((pred & (y == 0)).sum() / (y == 0).sum()), 4) if (y == 0).any() else None,
            "rappel_ia": round(float((pred & (y == 1)).sum() / (y == 1).sum()), 4) if (y == 1).any() else None,
            "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) == 2 else None,
        }
    return rows


def full_report(frame, prob, thresholds):
    return {
        "seuil_0_5": evaluate(frame["label"], prob, 0.5),
        "seuil_prudent_ia": evaluate(frame["label"], prob, thresholds["ai"]),
        "decisions": decision_summary(frame["label"], prob, thresholds),
        "par_source": per_source(frame, prob, 0.5),
    }


def leave_one_domain_out(config, tr, va, te):
    """Entraîne sans un domaine, l'évalue sur ce domaine (validation + test)."""
    held = pd.concat([va, te], ignore_index=True)
    results = {}
    if tr["source"].nunique() < 2:
        return results
    for source in sorted(set(tr["source"]) & set(held["source"])):
        tr_s = tr[tr["source"] != source]
        va_s = va[va["source"] != source]
        ev = held[held["source"] == source].reset_index(drop=True)
        if (ev["label"].nunique() < 2 or len(ev) < 20
                or tr_s["label"].nunique() < 2 or va_s["label"].nunique() < 2):
            continue
        det = Detector(build_pipeline(**config)).fit(tr_s["text"], tr_s["label"])
        det.calibrate(va_s["text"], va_s["label"])
        prob = det.predict_proba(ev["text"])[:, 1]
        m = evaluate(ev["label"], prob, 0.5)
        results[source] = {k: m[k] for k in ("roc_auc", "balanced_accuracy", "taux_faux_positifs", "rappel_ia", "nombre_test")}
    if results:
        results["moyenne_auc"] = round(float(np.mean([r["roc_auc"] for r in results.values()])), 4)
    return results


def train(data_dir=DATA_DIR, model_path=MODEL_PATH, meta_path=META_PATH, config_name="complet",
          lodo=True, verbose=True, recent=True):
    data_dir, model_path, meta_path = Path(data_dir), Path(model_path), Path(meta_path)
    frames = {split: load_partition(data_dir / f"{split}.csv")
              for split in ("train", "validation", "test")}
    # Corpus « LLM récents » (python -m detector.generate_data), s'il existe.
    if recent:
        for split in ("train", "validation"):
            extra = load_partition(data_dir / f"recent_{split}.csv", required=False)
            if extra is not None:
                frames[split] = pd.concat([frames[split], extra], ignore_index=True)
    for split in ("test_natif", "test_recent"):
        extra = load_partition(data_dir / f"{split}.csv", required=False)
        if extra is not None:  # évalué même avec --sans-recent, pour comparer
            frames[split] = extra

    # Les textes identiques ne doivent jamais figurer dans plusieurs partitions.
    split_sets = {k: set(v["text"].str.casefold()) for k, v in frames.items()}
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if split_sets[a] & split_sets[b]:
                raise ValueError(f"Fuite de données : doublons exacts entre {a} et {b}")

    config = dict(CONFIGS[config_name])
    if config.get("llm"):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            print("torch/transformers absents : entraînement sans les caractéristiques de perplexité.")
            config_name = "sans_llm" if config.get("words") else "style"
            config["llm"] = False

    tr, va, te = frames["train"], frames["validation"], frames["test"]
    if verbose:
        print(f"Configuration : {config_name} | " + " ; ".join(f"{k} {len(v)}" for k, v in frames.items()))
    detector = Detector(build_pipeline(**config), description=config_name)
    detector.fit(tr["text"], tr["label"])
    detector.calibrate(va["text"], va["label"])
    if detector.uses_llm:
        detector.record_llm_reference()
    thresholds = choose_thresholds(va["label"], detector.predict_proba(va["text"])[:, 1])

    metrics = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": f"{HF_DATASET} (hc3_fr, échantillonnage apparié par question)",
        "configuration": config_name,
        "caracteristiques": {k: bool(v) for k, v in config.items()},
        "method": describe(config),
        "labels": {"0": "humain", "1": "IA"},
        "size": {k: len(v) for k, v in frames.items()},
        "thresholds": thresholds,
    }
    for split in ("test", "test_natif", "test_recent"):
        if split in frames:
            prob = detector.predict_proba(frames[split]["text"])[:, 1]
            metrics[split] = full_report(frames[split], prob, thresholds)
            if "generateur" in frames[split].columns:
                metrics[split]["par_generateur"] = per_source(
                    frames[split].assign(source=frames[split]["generateur"].fillna("humain")), prob, 0.5)
    # Compatibilité avec l'ancienne interface (barre latérale).
    metrics["test_seuil_0_5"] = metrics["test"]["seuil_0_5"]
    metrics["test_seuil_prudent_ia"] = metrics["test"]["seuil_prudent_ia"]
    if lodo:
        if verbose:
            print("Évaluation hors domaine (un modèle par domaine retiré)...")
        metrics["hors_domaine"] = leave_one_domain_out(config, tr, va, te)
    metrics["limitations"] = [
        "Corpus HC3 français majoritairement traduit depuis l'anglais",
        "IA : ChatGPT/BingGPT (2022-2023) + Gemini 3.x (2026) ; autres LLM récents non testés",
        "Test natif : textes IA = réponses à des questions, textes humains = surtout des FAQ",
        "Le score est une sortie de modèle, pas une preuve d'auteur",
        "Les textes très courts et hors domaine sont particulièrement incertains",
    ]

    detector.disable_disk_cache()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, model_path)
    meta_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print_summary(metrics)
        print(f"Modèle : {model_path} | métriques : {meta_path}")
    return metrics


def describe(config):
    parts = []
    if config.get("words"):
        parts.append("TF-IDF mots (1-2)")
    if config.get("chars"):
        parts.append("TF-IDF caractères (2-4)")
    if config.get("style"):
        parts.append("stylométrie")
    if config.get("llm"):
        parts.append("perplexité / Binoculars (Qwen2.5-0.5B)")
    return " + ".join(parts) + " → régression logistique calibrée"


def print_summary(m):
    print(f"\nSeuils : humain <= {m['thresholds']['human']:.3f} | IA >= {m['thresholds']['ai']:.3f}")
    for split in ("test", "test_natif", "test_recent"):
        if split in m:
            s = m[split]["seuil_0_5"]
            print(f"{split:11s} AUC {s['roc_auc']}  bal.acc {s['balanced_accuracy']}  "
                  f"FPR {s['taux_faux_positifs']}  rappel IA {s['rappel_ia']}")
            for src, r in m[split]["par_source"].items():
                print(f"    {src:22s} n={r['n']:4d} AUC={r['roc_auc']} FPR={r['taux_faux_positifs']} rappel={r['rappel_ia']}")
    if "hors_domaine" in m:
        print("hors domaine :", {k: (v["roc_auc"] if isinstance(v, dict) else v) for k, v in m["hors_domaine"].items()})


def main():
    parser = argparse.ArgumentParser(description="Entraîne le détecteur local et évalue sa qualité.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--meta-path", type=Path, default=META_PATH)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="complet",
                        help="complet = TF-IDF + style + perplexité (nécessite requirements-llm.txt)")
    parser.add_argument("--sans-recent", action="store_true",
                        help="N'entraîne pas sur le corpus LLM récents (test_recent.csv reste évalué)")
    parser.add_argument("--sans-hors-domaine", action="store_true",
                        help="Saute l'évaluation leave-one-domain-out (plus rapide)")
    args = parser.parse_args()
    train(args.data_dir, args.model_path, args.meta_path, args.config, lodo=not args.sans_hors_domaine,
          recent=not args.sans_recent)


if __name__ == "__main__":
    main()
