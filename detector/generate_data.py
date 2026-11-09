"""Construit un corpus apparié « LLM récents » avec l'API Gemini.

Exécution : python -m detector.generate_data   (clé dans .env : GEMINI_API_KEY=...)

Principe (le même que pour HC3) : chaque texte IA est généré pour le MÊME sujet
qu'un texte humain écrit avant l'arrivée des LLM. Le modèle ne peut donc pas
distinguer les classes par le thème, seulement par la manière d'écrire.

Sources humaines (toutes antérieures à 2022) :
* questions/réponses HC3 (questions non utilisées par download_data) ;
* FAQ de sites français (hc3_french_ood/faq_fr_random) ;
* critiques de films Allociné (tblard/allocine, licence MIT).

Chaque génération reçoit un style tiré au hasard (neutre, forum, post
LinkedIn, étudiant, avis personnel, « écris comme un humain »...) et un modèle
Gemini tiré au hasard, pour éviter que le détecteur n'apprenne un seul style.

Sorties : data/recent_train.csv, data/recent_validation.csv, data/test_recent.csv
Les réponses de l'API sont mises en cache (data/raw/gemini_cache.jsonl) :
le script peut être interrompu et relancé sans tout regénérer.
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_DIR, PROJECT_DIR, RANDOM_SEED
from .download_data import clean_text, read_jsonl, text_hash

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash")
ALLOCINE_URL = "https://huggingface.co/datasets/tblard/allocine/resolve/main/allocine/{split}-00000-of-00001.parquet"

STYLES = {
    "neutre": "",
    "forum": "Réponds comme un internaute sur un forum, de façon naturelle et sans formules toutes faites.",
    "linkedin": "Adopte un ton inspirant avec des phrases courtes et percutantes, comme un post LinkedIn.",
    "etudiant": "Rédige comme un étudiant qui fait un devoir soigné.",
    "avis_perso": "Écris à la première personne, de façon décontractée, en donnant ton avis.",
    "humain": "Écris de façon à ce que le texte paraisse rédigé par un humain, pas par une IA.",
    "expert": "Réponds comme un expert du domaine, de façon précise et pédagogique.",
}
COMMON_RULES = ("Écris en français, en texte brut : pas de Markdown, pas de titre, pas de liste à puces, "
                "pas d'introduction du type « Voici ». Environ {words} mots.")


def load_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    env_file = PROJECT_DIR / ".env"
    if not key and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("Clé absente : ajoute GEMINI_API_KEY=... dans le fichier .env du projet.")
    return key


class GeminiClient:
    """Petit client avec cache disque, rotation de modèles et attente sur quota."""

    def __init__(self, key, models, cache_path, min_interval=4.5):
        self.key = key
        self.models = list(models)
        self.disabled = set()
        self.cache_path = Path(cache_path)
        self.cache = {}
        self.min_interval = min_interval
        self.last_call = defaultdict(float)
        if self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    self.cache[item["key"]] = item
        self.session = requests.Session()

    def generate(self, prompt, preferred_model, rng, variant=""):
        # `variant` distingue deux sujets dont le prompt est identique (critiques
        # Allociné) : sans lui, le cache renverrait deux fois le même texte.
        key = hashlib.sha256((prompt + variant).encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]["text"], self.cache[key]["model"]
        candidates = [preferred_model] + [m for m in self.models if m != preferred_model]
        for attempt in range(6):
            for model in candidates:
                if model in self.disabled:
                    continue
                wait = self.min_interval - (time.time() - self.last_call[model])
                if wait > 0:
                    time.sleep(wait)
                self.last_call[model] = time.time()
                try:
                    response = self.session.post(
                        API_URL.format(model=model), headers={"x-goog-api-key": self.key}, timeout=90,
                        json={"contents": [{"parts": [{"text": prompt}]}],
                              "generationConfig": {"temperature": rng.choice([0.7, 1.0, 1.2])}},
                    )
                except requests.RequestException:
                    continue
                if response.status_code == 404:
                    self.disabled.add(model)
                    continue
                if response.status_code in (429, 500, 503):
                    if "PerDay" in response.text or "per day" in response.text.lower():
                        print(f"  quota journalier atteint pour {model}")
                        self.disabled.add(model)
                    continue
                if not response.ok:
                    continue
                parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                if not text:
                    continue
                item = {"key": key, "model": model, "text": text}
                self.cache[key] = item
                with open(self.cache_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                return text, model
            if all(m in self.disabled for m in self.models):
                raise RuntimeError("Tous les modèles ont atteint leur quota. Relance demain : le cache est conservé.")
            time.sleep(min(60, 10 * (attempt + 1)))
        return None, None


def strip_markdown(text):
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.M)
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)
    return text.strip()


def hc3_questions(raw_dir, used_groups, used_texts, rng, per_source):
    """Questions HC3 (split val) non utilisées, avec leur réponse humaine."""
    by_source = defaultdict(list)
    seen = set()
    for record in read_jsonl(raw_dir / "hc3_fr" / "val_qa_examples.json"):
        group = f"{record['source']}:{record['id']}"
        if group in used_groups or group in seen or record["is_human"] not in ("True", True):
            continue
        human = clean_text(record.get("answer"))
        question = (record.get("question") or "").strip()
        # HC3 contient des réponses identiques sous des identifiants différents.
        if human and question and text_hash(human) not in used_texts:
            used_texts.add(text_hash(human))
            seen.add(group)
            by_source[record["source"]].append({"group": group, "question": question,
                                                "human": human, "source": f"hc3/{record['source']}"})
    items = []
    for source in sorted(by_source):
        rng.shuffle(by_source[source])
        items += by_source[source][:per_source]
    return items


def faq_questions(raw_dir, used_texts, rng, limit):
    items = []
    for i, record in enumerate(read_jsonl(raw_dir / "faq_fr_random.jsonl")):
        human = clean_text(record.get("answer"))
        question = (record.get("question") or "").strip()
        if human and question and text_hash(human) not in used_texts:
            items.append({"group": f"faq:{i}", "question": question, "human": human, "source": "faq_web"})
    rng.shuffle(items)
    return items[:limit]


def allocine_reviews(cache_dir, rng, limit):
    path = cache_dir / "allocine_train.parquet"
    if not path.is_file():
        print("Téléchargement des critiques Allociné (~40 Mo)...")
        response = requests.get(ALLOCINE_URL.format(split="train"), timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    frame = pd.read_parquet(path)
    items = []
    for i, row in frame.sample(n=min(len(frame), limit * 3), random_state=RANDOM_SEED).iterrows():
        human = clean_text(row["review"])
        if human:
            items.append({"group": f"allocine:{i}", "polarity": "positive" if row["label"] == 1 else "négative",
                          "human": human, "source": "allocine"})
    return items[:limit]


def build_prompt(item, style_key, words):
    rules = COMMON_RULES.format(words=words)
    style = STYLES[style_key]
    if "question" in item:
        return f"{style} {rules}\n\nQuestion : {item['question']}".strip()
    return (f"Écris une critique de film {item['polarity']}, comme un spectateur qui publie son avis "
            f"sur un site de cinéma, à propos d'un film de ton choix. {style} {rules}").strip()


def generate_corpus(output_dir=DATA_DIR, models=DEFAULT_MODELS, per_source_hc3=60, faq=300, allocine=300,
                    min_interval=4.5):
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw" / "data"
    if not raw_dir.is_dir():
        raise SystemExit("Lance d'abord : python -m detector.download_data")
    rng = random.Random(RANDOM_SEED)
    used_groups, used_texts = set(), set()
    for split in ("train", "validation", "test", "test_natif"):
        path = output_dir / f"{split}.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            used_groups.update(frame.get("group", pd.Series(dtype=str)).dropna())
            used_texts.update(frame["text"].map(text_hash))

    items = (hc3_questions(raw_dir, used_groups, used_texts, rng, per_source_hc3)
             + faq_questions(raw_dir, used_texts, rng, faq)
             + allocine_reviews(output_dir / "raw", rng, allocine))
    rng.shuffle(items)
    client = GeminiClient(load_api_key(), models, output_dir / "raw" / "gemini_cache.jsonl", min_interval)
    print(f"{len(items)} sujets à traiter ({len(client.cache)} réponses déjà en cache).")

    rows = []
    for n, item in enumerate(items, 1):
        style = rng.choice(list(STYLES))
        words = max(60, min(400, len(item["human"].split())))
        model = rng.choice(list(models))
        prompt = build_prompt(item, style, words)
        variant = "" if "question" in item else f"|{item['group']}"
        text, used_model = client.generate(prompt, model, rng, variant)
        ai_text = clean_text(strip_markdown(text)) if text else None
        if ai_text is None:
            continue
        base = {"source": item["source"], "group": item["group"], "corpus": "llm_recent"}
        rows.append({**base, "text": item["human"], "label": 0, "generateur": "humain", "style": ""})
        rows.append({**base, "text": ai_text, "label": 1, "generateur": used_model, "style": style})
        if n % 25 == 0:
            print(f"  {n}/{len(items)} sujets")

    frame = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    # Découpage par sujet : les deux textes d'une paire restent ensemble.
    groups = sorted(frame["group"].unique())
    random.Random(RANDOM_SEED).shuffle(groups)
    n_val, n_test = int(len(groups) * 0.15), int(len(groups) * 0.25)
    assignment = {g: "recent_validation" for g in groups[:n_val]}
    assignment.update({g: "test_recent" for g in groups[n_val:n_val + n_test]})
    frame["split"] = frame["group"].map(assignment).fillna("recent_train")
    for split, part in frame.groupby("split"):
        part = part.drop(columns="split").sample(frac=1, random_state=RANDOM_SEED)
        part.to_csv(output_dir / f"{split}.csv", index=False, encoding="utf-8")
        print(f"{split}.csv : {len(part)} textes | modèles : "
              f"{part[part.label == 1]['generateur'].value_counts().to_dict()}")
    return frame


def main():
    parser = argparse.ArgumentParser(description="Génère un corpus apparié de textes de LLM récents (Gemini).")
    parser.add_argument("--per-source-hc3", type=int, default=60, help="Questions HC3 par domaine")
    parser.add_argument("--faq", type=int, default=300)
    parser.add_argument("--allocine", type=int, default=300)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--min-interval", type=float, default=4.5,
                        help="Secondes minimum entre deux appels à un même modèle (quota gratuit)")
    args = parser.parse_args()
    generate_corpus(models=args.models, per_source_hc3=args.per_source_hc3, faq=args.faq,
                    allocine=args.allocine, min_interval=args.min_interval)


if __name__ == "__main__":
    main()
