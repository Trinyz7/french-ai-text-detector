"""Télécharge le corpus HC3 français et construit des partitions équilibrées.

Exécution : python -m detector.download_data

Pourquoi ce script fait plus qu'un simple échantillonnage :

* **Échantillonnage apparié** : pour chaque question retenue, on garde une
  réponse humaine ET une réponse IA. Le sujet est donc identique dans les deux
  classes et le modèle ne peut plus « tricher » en reconnaissant le thème
  (ex. finance → IA, reddit → humain).
* **Équilibre par domaine** : chaque source (reddit, finance, medicine, open,
  wiki) reçoit le même quota, pour que l'évaluation couvre tous les domaines.
* **Test « natif »** (`test_natif.csv`) : textes écrits directement en
  français (ChatGPT, BingGPT, FAQ de sites français), jamais vus à
  l'entraînement. Il mesure la robustesse hors du corpus traduit.

Le jeu de données (CC BY-SA 4.0) est téléchargé une seule fois dans data/raw/.
"""

import argparse
import hashlib
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_DIR, HF_ARCHIVE_URL, RANDOM_SEED

RAW_DIR_NAME = "raw"
ARCHIVE_NAME = "hc3_french_ood.tar.gz"
MIN_WORDS = 35
MAX_CHARS = 50_000
SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}


def normalize_label(value):
    """Convention du projet : humain=0, IA=1."""
    if isinstance(value, bool):
        return 0 if value else 1  # HC3 : is_human=True → humain
    if value in (0, "0", "HUMAN", "human", "Human", "Humain", "True"):
        return 0
    if value in (1, "1", "CHATGPT", "chatgpt", "ChatGPT", "BingGPT", "AI", "ia", "False"):
        return 1
    raise ValueError(f"Étiquette inattendue : {value!r}")


def clean_text(text):
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    if len(text.split()) < MIN_WORDS or len(text) > MAX_CHARS:
        return None
    return text


def text_hash(text):
    return hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()


def download_archive(raw_dir, force=False):
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / ARCHIVE_NAME
    if archive.is_file() and not force:
        print(f"Archive déjà présente : {archive}")
    else:
        print(f"Téléchargement de {HF_ARCHIVE_URL} (~175 Mo) ...")
        tmp = archive.with_suffix(".part")
        with requests.get(HF_ARCHIVE_URL, stream=True, timeout=60,
                          headers={"User-Agent": "french-ai-text-detector/0.2"}) as response:
            response.raise_for_status()
            with open(tmp, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
        tmp.replace(archive)
    # Extraction des seuls fichiers français utiles, sans chemins dangereux.
    wanted = {
        "data/hc3_fr/train_qa_examples.json",
        "data/hc3_fr/val_qa_examples.json",
        "data/hc3_fr/test_qa_examples.json",
        "data/hc3_fr_chatgpt/test_qa_examples_chatgpt.jsonl",
        "data/qa_binggpt.jsonl",
        "data/qa_binglikehuman.jsonl",
        "data/faq_fr_gouv.jsonl",
        "data/faq_fr_random.jsonl",
    }
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name in wanted and m.isfile()]
        missing = [name for name in wanted if not (raw_dir / name).is_file()]
        if missing:
            tar.extractall(raw_dir, members=members, filter="data")
    return raw_dir / "data"


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_sample(records, per_source, rng, corpus):
    """Garde `per_source` questions par domaine, avec 1 réponse humaine + 1 IA chacune."""
    by_question = defaultdict(lambda: {0: [], 1: []})
    for record in records:
        text = clean_text(record.get("answer"))
        if text is None:
            continue
        label = normalize_label(record["is_human"])
        key = (record["source"], str(record["id"]))
        by_question[key][label].append(text)
    by_source = defaultdict(list)
    for (source, qid), answers in by_question.items():
        if answers[0] and answers[1]:
            by_source[source].append((qid, answers))
    rows = []
    for source in sorted(by_source):
        questions = sorted(by_source[source])
        rng.shuffle(questions)
        for qid, answers in questions[:per_source]:
            for label in (0, 1):
                rows.append({
                    "text": rng.choice(answers[label]),
                    "label": label,
                    "source": source,
                    "group": f"{source}:{qid}",
                    "corpus": corpus,
                })
    return rows


def native_test_set(data_dir, rng, per_class):
    """Textes rédigés directement en français (non traduits)."""
    ai, human = [], []
    for record in read_jsonl(data_dir / "hc3_fr_chatgpt" / "test_qa_examples_chatgpt.jsonl"):
        answers = record.get("chatgpt_answer") or []
        if isinstance(answers, str):
            answers = [answers]
        for answer in answers[:1]:
            text = clean_text(answer)
            if text:
                ai.append({"text": text, "label": 1, "source": f"chatgpt_fr/{record['source']}",
                           "group": f"natif:chatgpt:{record['source']}:{record['id']}",
                           "corpus": "natif"})
    for record in read_jsonl(data_dir / "qa_binggpt.jsonl"):
        text = clean_text(record.get("answer"))
        if text:
            ai.append({"text": text, "label": 1, "source": "binggpt",
                       "group": f"natif:bing:{record['id']}", "corpus": "natif"})
    for name, source in (("qa_binglikehuman.jsonl", "web_humain"),
                         ("faq_fr_gouv.jsonl", "faq_gouv"),
                         ("faq_fr_random.jsonl", "faq_web")):
        for i, record in enumerate(read_jsonl(data_dir / name)):
            text = clean_text(record.get("answer"))
            if text:
                human.append({"text": text, "label": 0, "source": source,
                              "group": f"natif:{source}:{i}", "corpus": "natif"})
    rng.shuffle(ai)
    rng.shuffle(human)
    # Les sous-corpus rares (BingGPT, humain « style Bing ») sont tous gardés.
    def take(pool, rare):
        keep = [r for r in pool if r["source"] in rare]
        rest = [r for r in pool if r["source"] not in rare]
        return keep + rest[: max(0, per_class - len(keep))]
    return take(ai, {"binggpt"}) + take(human, {"web_humain", "faq_gouv"})


def build_datasets(output_dir=DATA_DIR, per_source_train=300, per_source_validation=60,
                   per_source_test=50, native_per_class=300, force_download=False):
    output_dir = Path(output_dir)
    data_dir = download_archive(output_dir / RAW_DIR_NAME, force=force_download)
    rng = random.Random(RANDOM_SEED)
    quotas = {"train": per_source_train, "validation": per_source_validation, "test": per_source_test}
    frames = {}
    for split, file_prefix in SPLIT_FILES.items():
        records = read_jsonl(data_dir / "hc3_fr" / f"{file_prefix}_qa_examples.json")
        frames[split] = pd.DataFrame(paired_sample(records, quotas[split], rng, "hc3_fr_traduit"))
    frames["test_natif"] = pd.DataFrame(native_test_set(data_dir, rng, native_per_class))

    # Aucun texte ni aucune question ne doit apparaître dans deux partitions.
    seen_texts, seen_groups = set(), set()
    for split in ("train", "validation", "test", "test_natif"):
        frame = frames[split]
        hashes = frame["text"].map(text_hash)
        keep = ~hashes.isin(seen_texts) & ~frame["group"].isin(seen_groups)
        frame = frame[keep].drop_duplicates(subset=["text"]).reset_index(drop=True)
        seen_texts.update(frame["text"].map(text_hash))
        seen_groups.update(frame["group"])
        counts = frame["label"].value_counts().to_dict()
        if counts.get(0, 0) < 10 or counts.get(1, 0) < 10:
            raise ValueError(f"Pas assez de données ({split}) : {counts}")
        frame = frame.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        frames[split] = frame
        target = output_dir / f"{split}.csv"
        frame.to_csv(target, index=False, encoding="utf-8")
        by_source = frame.groupby(["source", "label"]).size().unstack(fill_value=0)
        print(f"\n{target.name} : {len(frame)} textes\n{by_source.to_string()}")
    print("\nDonnées prêtes (licence du corpus : CC BY-SA 4.0).")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Prépare le corpus français humain / IA.")
    parser.add_argument("--per-source-train", type=int, default=300,
                        help="Questions par domaine pour l'entraînement (x2 textes)")
    parser.add_argument("--per-source-validation", type=int, default=60)
    parser.add_argument("--per-source-test", type=int, default=50)
    parser.add_argument("--native-per-class", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    build_datasets(args.output_dir, args.per_source_train, args.per_source_validation,
                   args.per_source_test, args.native_per_class, args.force_download)


if __name__ == "__main__":
    main()
