"""Télécharge un échantillon équilibré du jeu HC3 français via l'API publique HF.

Exécution : python -m detector.download_data --train-size 2000
Le téléchargement ne nécessite ni compte ni clé API, mais nécessite Internet.
"""

import argparse
import hashlib
import random
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import DATA_DIR, HF_DATASET, HF_ENDPOINT, HF_SUBSET, RANDOM_SEED

BATCH_SIZE = 100  # Limite officielle de l'API datasets-server


def normalize_label(value):
    """Convention HC3: humain=0, IA=1."""
    if isinstance(value, bool):
        raise ValueError("Label booléen ambigu")
    if value in (0, "0", "HUMAN", "human", "Humain"):
        return 0
    if value in (1, "1", "CHATGPT", "chatgpt", "AI", "ia"):
        return 1
    raise ValueError(f"Étiquette inattendue : {value!r}")


def make_session():
    session = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "french-ai-text-detector/0.1 (educational project)"})
    return session


def get_page(session, split, offset, length=BATCH_SIZE):
    params = {
        "dataset": HF_DATASET,
        "config": HF_SUBSET,
        "split": split,
        "offset": offset,
        "length": length,
    }
    response = session.get(HF_ENDPOINT, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if "rows" not in payload or "num_rows_total" not in payload:
        raise ValueError(f"Réponse HF inattendue pour {split}: {list(payload)}")
    return payload


def sample_split(session, split, target_size, rng):
    if target_size < 20 or target_size % 2:
        raise ValueError("La taille demandée doit être paire et >= 20")
    # Échantillonnage aléatoire de pages sur toute la partition,
    # et non les N premières lignes (potentiellement triées par label).
    first = get_page(session, split, 0, 1)
    total = int(first["num_rows_total"])
    offsets = list(range(0, total, BATCH_SIZE))
    rng.shuffle(offsets)
    quota = target_size // 2
    samples = {0: [], 1: []}
    seen_texts = set()
    for index, offset in enumerate(offsets, 1):
        payload = get_page(session, split, offset)
        rows = [entry["row"] for entry in payload["rows"]]
        rng.shuffle(rows)
        for record in rows:
            try:
                label = normalize_label(record["label"])
            except (ValueError, KeyError):
                continue
            text = record.get("answer")
            if not isinstance(text, str):
                continue
            text = " ".join(text.split())
            if len(text.split()) < 35 or len(text) > 50_000:
                continue
            digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
            if digest in seen_texts or len(samples[label]) >= quota:
                continue
            seen_texts.add(digest)
            samples[label].append({
                "text": text,
                "label": label,
                "source": str(record.get("source", "")),
                "origin_id": str(record.get("id", "")),
            })
        if index % 15 == 0:
            print(f"  {split}: humain={len(samples[0])}/{quota}, IA={len(samples[1])}/{quota}")
        if all(len(group) >= quota for group in samples.values()):
            break
        time.sleep(0.07)  # Réduit le rythme des requêtes publiques.
    if min(map(len, samples.values())) < quota:
        raise RuntimeError(
            f"Données insuffisantes dans '{split}': humain={len(samples[0])}, "
            f"IA={len(samples[1])}; demandé {quota} par classe. "
            "Relance avec une taille plus petite ou réessaie plus tard."
        )
    combined = samples[0] + samples[1]
    rng.shuffle(combined)
    return pd.DataFrame(combined)


def fetch_all(output_dir=DATA_DIR, train_size=2000, validation_size=400, test_size=400):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    session = make_session()
    frames = {}
    # Splits déjà définis par les auteurs : pas de mélange aléatoire global.
    for split, size in (
        ("train", train_size),
        ("validation", validation_size),
        ("test", test_size),
    ):
        print(f"Téléchargement {split}: {size} exemples (50 % humains / 50 % IA) ...")
        frames[split] = sample_split(session, split, size, rng)
    # Supprime les éventuels doublons exacts entre partitions.
    seen = set()
    for split in ("train", "validation", "test"):
        frame = frames[split]
        hashes = frame["text"].str.casefold().map(
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
        keep = ~hashes.isin(seen)
        frames[split] = frame[keep].reset_index(drop=True)
        seen.update(hashes[keep])
        counts = frames[split]["label"].value_counts().to_dict()
        if not (counts.get(0, 0) >= 10 and counts.get(1, 0) >= 10):
            raise ValueError(f"Pas assez de données après dédoublonnage ({split}): {counts}")
        target = output_dir / f"{split}.csv"
        frames[split].to_csv(target, index=False, encoding="utf-8")
        print(f"  Sauvegardé : {target} ({len(frames[split])} lignes, classes={counts})")
    print("Données prêtes. N'ajoute pas les CSV téléchargés au dépôt GitHub sans vérifier les licences.")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Télécharge le corpus français humain / IA.")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    fetch_all(args.output_dir, args.train_size, args.validation_size, args.test_size)


if __name__ == "__main__":
    main()
