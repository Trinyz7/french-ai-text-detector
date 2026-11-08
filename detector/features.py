"""Caractéristiques indépendantes du sujet : style d'écriture et perplexité.

Les TF-IDF captent beaucoup le *thème* d'un texte. Les caractéristiques ci-dessous
cherchent plutôt *comment* il est écrit, ce qui généralise mieux d'un domaine
à l'autre.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .config import CACHE_DIR, LM_MAX_TOKENS, LM_OBSERVER, LM_PERFORMER

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœ]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿŒœ]+)*")
SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]+|[^.!?…]+$")

# Connecteurs et formules très fréquents dans les réponses de LLM en français.
LLM_PHRASES = (
    "cependant", "en outre", "de plus", "par conséquent", "en conclusion",
    "en résumé", "il est important", "il est essentiel", "il convient",
    "néanmoins", "toutefois", "en effet", "notamment", "globalement",
    "en général", "en d'autres termes", "il est possible", "il existe",
    "plusieurs facteurs", "dans l'ensemble", "il est également", "généralement",
)
HEDGES = ("peut-être", "je pense", "je crois", "probablement", "à mon avis",
          "j'imagine", "sûrement", "je suppose", "il me semble")
INFORMAL = ("lol", "mdr", "ouais", "bah", "genre", "truc", "trucs", "ça", "hein",
            "ok", "bref", "perso", "super", "vraiment", "carrément", "tellement")
FIRST_PERSON = {"je", "j", "me", "m", "moi", "mon", "ma", "mes", "nous", "notre", "nos"}
SECOND_PERSON = {"tu", "te", "toi", "ton", "ta", "tes", "vous", "votre", "vos"}
STOPWORDS = set("""le la les un une des de du d l et à au aux en dans pour par sur
avec ce cette ces qui que qu est sont a ont il elle ils elles on ne pas plus se
sa son ses leur leurs y ou mais donc car si comme être avoir fait""".split())

STYLE_FEATURE_NAMES = [
    "log_nb_mots", "longueur_moy_mot", "richesse_vocab_mattr", "taux_hapax",
    "longueur_moy_phrase", "ecart_type_phrase", "variation_phrases",
    "virgules", "points_virgules", "deux_points", "exclamations", "interrogations",
    "parentheses", "guillemets", "points_suspension", "tirets", "chiffres",
    "majuscules", "premiere_personne", "deuxieme_personne", "connecteurs_llm",
    "hesitations", "familier", "mots_outils", "repetition_bigrammes",
    "debut_phrase_minuscule", "espaces_typo_fr",
]


def _rate(count, total):
    return count / total if total else 0.0


def _count_phrases(lower, phrases):
    return sum(len(re.findall(r"(?<!\w)" + re.escape(p) + r"(?!\w)", lower)) for p in phrases)


def _mattr(words, window=50):
    """Moving-average type/token ratio : richesse lexicale peu sensible à la longueur."""
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    ratios = [len(set(words[i:i + window])) / window
              for i in range(0, len(words) - window + 1, max(1, window // 5))]
    return float(np.mean(ratios))


def style_features(text):
    text = " ".join(str(text).split())
    lower = text.lower()
    words = [w.lower().replace("’", "'") for w in WORD_RE.findall(text)]
    n = len(words)
    tokens = [t for w in words for t in re.split(r"'", w) if t]
    sentences = [s.strip() for s in SENTENCE_RE.findall(text) if len(s.split()) > 0]
    lengths = np.array([len(WORD_RE.findall(s)) for s in sentences] or [0], dtype=float)
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    bigrams = list(zip(words, words[1:]))
    letters = sum(c.isalpha() for c in text)
    mean_len = float(lengths.mean())
    std_len = float(lengths.std())
    return [
        math.log1p(n),
        float(np.mean([len(w) for w in words])) if words else 0.0,
        _mattr(words),
        _rate(sum(1 for c in counts.values() if c == 1), n),
        mean_len,
        std_len,
        std_len / mean_len if mean_len else 0.0,
        _rate(text.count(","), n),
        _rate(text.count(";"), n),
        _rate(text.count(":"), n),
        _rate(text.count("!"), n),
        _rate(text.count("?"), n),
        _rate(text.count("("), n),
        _rate(sum(text.count(q) for q in '«»"“”'), n),
        _rate(text.count("...") + text.count("…"), n),
        _rate(text.count(" - ") + text.count(" – "), n),
        _rate(sum(c.isdigit() for c in text), len(text)),
        _rate(sum(c.isupper() for c in text), letters),
        _rate(sum(t in FIRST_PERSON for t in tokens), n),
        _rate(sum(t in SECOND_PERSON for t in tokens), n),
        _rate(_count_phrases(lower, LLM_PHRASES), n) * 100,
        _rate(_count_phrases(lower, HEDGES), n) * 100,
        _rate(_count_phrases(lower, INFORMAL), n) * 100,
        _rate(sum(t in STOPWORDS for t in tokens), len(tokens)),
        _rate(len(bigrams) - len(set(bigrams)), len(bigrams)),
        _rate(sum(1 for s in sentences if s[:1].islower()), len(sentences)),
        _rate(len(re.findall(r"\s[;:!?»]", text)), n),
    ]


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Statistiques stylométriques (ponctuation, rythme des phrases, vocabulaire...)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([style_features(t) for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"style:{n}" for n in STYLE_FEATURE_NAMES])


LM_FEATURE_NAMES = [
    "perplexite_log", "perplexite_log_instruit", "entropie_croisee",
    "score_binoculars", "variabilite_perte", "part_tokens_top1",
]


class PerplexityFeatures(BaseEstimator, TransformerMixin):
    """Mesures calculées par un petit modèle de langue local (méthode Binoculars).

    Un texte généré par IA est en général *moins surprenant* pour un modèle de
    langue (perplexité basse) et de façon plus *régulière*. Binoculars compare
    la perplexité d'un modèle à l'entropie croisée entre deux modèles proches,
    ce qui corrige l'effet « sujet difficile ».

    Les modèles sont chargés à la demande et ne sont jamais enregistrés dans le
    fichier joblib. Les résultats sont mis en cache sur disque par texte.
    """

    def __init__(self, observer=LM_OBSERVER, performer=LM_PERFORMER,
                 max_tokens=LM_MAX_TOKENS, batch_size=8, cache_dir=CACHE_DIR):
        self.observer = observer
        self.performer = performer
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.cache_dir = cache_dir

    # --- sérialisation : on n'enregistre pas les poids des modèles ---
    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_models", "_cache", "_dirty"):
            state.pop(key, None)
        return state

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array([f"llm:{n}" for n in LM_FEATURE_NAMES])

    def _cache_path(self):
        if self.cache_dir is None:
            return None
        name = hashlib.sha1(f"{self.observer}|{self.performer}|{self.max_tokens}".encode()).hexdigest()[:12]
        return Path(self.cache_dir) / f"llm_features_{name}.json"

    def _load_cache(self):
        if getattr(self, "_cache", None) is None:
            self._cache, self._dirty = {}, False
            path = self._cache_path()
            if path and path.is_file():
                try:
                    self._cache = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    self._cache = {}
        return self._cache

    def _save_cache(self):
        path = self._cache_path()
        if path and getattr(self, "_dirty", False):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            tmp.replace(path)
            self._dirty = False

    def _load_models(self):
        if getattr(self, "_models", None) is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - dépend de l'installation
                raise ImportError(
                    "Les caractéristiques de perplexité nécessitent torch et transformers : "
                    "pip install -r requirements-llm.txt"
                ) from exc
            torch.set_grad_enabled(False)
            tokenizer = AutoTokenizer.from_pretrained(self.observer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            obs = AutoModelForCausalLM.from_pretrained(self.observer, dtype=torch.float32).eval()
            perf = AutoModelForCausalLM.from_pretrained(self.performer, dtype=torch.float32).eval()
            self._models = (torch, tokenizer, obs, perf)
        return self._models

    def _compute(self, texts):
        torch, tokenizer, obs, perf = self._load_models()
        enc = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True,
                        max_length=self.max_tokens)
        obs_logits = obs(**enc).logits[:, :-1].float()
        perf_logits = perf(**enc).logits[:, :-1].float()
        target = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        obs_logp = torch.log_softmax(obs_logits, dim=-1)
        perf_p = torch.softmax(perf_logits, dim=-1)
        perf_logp = torch.log_softmax(perf_logits, dim=-1)
        obs_nll = -obs_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        perf_nll = -perf_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        x_ent = -(perf_p * obs_logp).sum(-1)
        top1 = (obs_logits.argmax(-1) == target).float()
        rows = []
        for i in range(target.shape[0]):
            m = mask[i]
            k = m.sum().clamp(min=1)
            nll = (obs_nll[i] * m).sum() / k
            pnll = (perf_nll[i] * m).sum() / k
            xent = (x_ent[i] * m).sum() / k
            var = (((obs_nll[i] - nll) ** 2) * m).sum() / k
            rows.append([
                float(nll), float(pnll), float(xent),
                float(nll / xent.clamp(min=1e-6)),
                float(var.sqrt()), float((top1[i] * m).sum() / k),
            ])
        return rows

    def transform(self, X):
        texts = [" ".join(str(t).split()) for t in X]
        # Sans cache_dir (cas de l'application), rien n'est conservé :
        # les textes analysés ne sont ni écrits sur disque ni gardés en mémoire.
        cache = self._load_cache() if self.cache_dir is not None else {}
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        todo = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items(),
                      key=lambda kv: len(kv[1]))  # lots de longueur proche = moins de padding
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            for (key, _), row in zip(chunk, self._compute([t for _, t in chunk])):
                cache[key] = row
            self._dirty = True
            if len(todo) > 50 and (start // self.batch_size) % 25 == 0:
                print(f"  perplexité : {min(start + self.batch_size, len(todo))}/{len(todo)} textes")
                self._save_cache()
        self._save_cache()
        return np.array([cache[k] for k in keys], dtype=float)
