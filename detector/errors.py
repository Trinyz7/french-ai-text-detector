"""Repérage simple des fautes d'orthographe et de grammaire en français.

Un LLM ne fait presque jamais de faute : un texte qui en contient est donc
plutôt écrit (ou retouché) par un humain. Ce module ne remplace pas un vrai
correcteur ; il repère :

* les fautes d'orthographe typiques : accent oublié ou erroné (« tete »,
  « succés »), lettre doublée en trop ou manquante, lettres inversées ; un
  mot technique ou étranger absent du dictionnaire ne compte pas ;
* quelques fautes d'accord faciles à vérifier sans analyse grammaticale
  complète (« une crise majeur », « les problèmes économique ») ;
* quelques confusions classiques (« il à été », « il a manger », « ils mange »).

Les règles sont volontairement prudentes : mieux vaut rater une faute que
signaler une phrase correcte. Les fautes trouvées font baisser le score IA
(voir ``human_evidence_factor``).
"""

import re
from functools import lru_cache

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿŒœÆæ]+)*")
ELISIONS = {"l", "d", "j", "m", "n", "s", "t", "c", "qu", "jusqu", "lorsqu", "puisqu", "quoiqu"}
SPECIAL_WORDS = {"aujourd'hui", "quelqu'un", "quelqu'une", "presqu'île", "prud'homme",
                 "prud'hommes", "entr'ouvert"}

# Mots grammaticaux (jamais des adjectifs à accorder).
FUNCTION_WORDS = set("""
à a ai as au aux avec bien car ce ceci cela ces cet cette chez comme comment dans de des
dont du elle elles en encore entre est et été être eu fait faut il ils je jamais la le les
leur leurs lui mais mal même mêmes moi moins mon ne ni non nous on ont ou où par pas peu
plus pour pourquoi quand que quel quelle qui quoi sa sans se ses si son sont sous sur ta
tant te tes toi ton tous tout toute toutes très trop tu un une vers vous y genre déjà aussi
après avant ici là puis alors donc depuis pendant parce lors selon contre malgré hors
ensemble autre autres presque derrière devant environ d l j m n s t c qu
""".split())

# Terminaisons d'adjectifs / participes au masculin singulier qui prennent « e »
# au féminin et « s » au pluriel (majeur, dernier, effectué, normal...).
# (Pas -ant/-ent : participes présents invariables, « les personnes vivant ».)
ADJ_MASC_RE = re.compile(r"(?:é|al|el|if|eur|ois|ais|ain|ien|ard|ier|eil)$")
# Adjectifs déjà terminés par « e » au singulier (économique, simple...).
ADJ_E_RE = re.compile(r"(?:ique|ible|able|aire|ive|euse|ée|imple|ile|ide)$")
FEMININE_DETERMINERS = {"une", "la", "cette"}
PLURAL_DETERMINERS = {"les", "des", "ces", "mes", "tes", "ses", "nos", "vos", "leurs",
                      "plusieurs", "quelques", "certains", "certaines", "nombreux", "nombreuses"}
SUBJECTS = {"il", "elle", "on", "qui", "ils", "elles", "nous", "vous", "j", "je", "tu", "ça", "cela"}
AVOIR = {"a", "ai", "as", "avons", "avez", "ont", "avait", "avais", "avaient"}
MODALS = {"peut", "peux", "veut", "veux", "doit", "dois", "doivent", "faut", "va", "vais",
          "vont", "peuvent", "veulent", "pour", "sans"}
AFTER_A = {"été", "eu", "fait", "pu", "dû", "dit", "vu", "mis", "pris", "un", "une", "des",
           "pas", "toujours", "jamais", "beaucoup"}

ERROR_LABELS = {
    "orthographe": "faute d'orthographe (accent, lettre doublée ou inversée)",
    "accord": "accord en genre ou en nombre",
    "confusion": "confusion fréquente (a/à, infinitif/participe...)",
}

# Fréquence minimale du mot correct supposé : on ne « corrige » que vers un mot courant.
COMMON_FREQUENCY = 1e-6
ACCENTS = str.maketrans("àâäçéèêëîïôöùûüÿ", "aaaceeeeiioouuuy")


def _normalize(word):
    return word.replace("œ", "oe").replace("æ", "ae")


@lru_cache(maxsize=1)
def _dictionary():
    """(mots, index sans accents, total), ou None si pyspellchecker n'est pas installé."""
    try:
        from spellchecker import SpellChecker
    except ImportError:
        return None
    words = dict(SpellChecker(language="fr", distance=1).word_frequency.dictionary)
    total = float(sum(words.values()))
    stripped = {}
    for word, freq in words.items():
        key = word.translate(ACCENTS)
        if freq > stripped.get(key, (None, 0))[1]:
            stripped[key] = (word, freq)
    return words, stripped, total


def _known(word, words):
    word = _normalize(word)
    if word in words:
        return True
    # Rectifications de 1990 : « entrainer », « connaitre » sont admis.
    for plain, hat in (("i", "î"), ("u", "û")):
        for k, c in enumerate(word):
            if c == plain and word[:k] + hat + word[k + 1:] in words:
                return True
    return False


def _is_typo(word, loaded):
    """Vrai si le mot est inconnu mais ressemble à un mot courant mal écrit.

    On ne retient que des fautes typiques d'un humain : accents oubliés ou
    erronés (« tete », « succés »), lettre doublée en trop ou manquante
    (« achetter », « acumulé ») et lettres inversées. Un mot simplement
    absent du dictionnaire (terme technique, mot étranger) ne compte pas.
    """
    words, stripped, total = loaded
    word = _normalize(word)
    if len(word) < 4 or _known(word, words):
        return False
    # Participe d'un verbe connu absent du dictionnaire (« domestiqués »).
    stem = re.sub(r"(?:é|ée|és|ées)$", "", word)
    if stem != word and _known(stem + "er", words):
        return False
    common = lambda w: words.get(w, 0) / total >= COMMON_FREQUENCY
    match = stripped.get(word.translate(ACCENTS))
    if match and match[0] != word and match[1] / total >= COMMON_FREQUENCY:
        return True
    if len(word) < 6:
        return False
    for i in range(len(word) - 1):
        if word[i] == word[i + 1] and common(word[:i] + word[i + 1:]):
            return True  # lettre doublée en trop
        if word[i] != word[i + 1] and common(word[:i] + word[i + 1] + word[i] + word[i + 2:]):
            return True  # lettres inversées
    for i in range(1, len(word)):
        if word[i].isalpha() and common(word[:i] + word[i] + word[i:]):
            return True  # lettre doublée manquante
    return False


def _adjective(w, words, feminine=False):
    """Vrai pour un adjectif variable au masculin singulier (pas un nom ni un verbe)."""
    if w.endswith("e"):
        return (not feminine and bool(ADJ_E_RE.search(w)) and _known(w + "s", words)
                and not _known(w[:-1] + "er", words))
    if not ADJ_MASC_RE.search(w):
        return False
    return _known(w, words) and _known(w + "e", words) and _known(w + "s", words)


def _split_elisions(token):
    """« d'apprentissage » -> ["d'", "apprentissage"] ; « aujourd'hui » reste entier."""
    word = token.lower().replace("’", "'")
    if word in SPECIAL_WORDS or "'" not in word:
        return [word]
    head, _, rest = word.partition("'")
    return [head + "'"] + _split_elisions(rest) if head in ELISIONS else [word]


def _spelling_errors(tokens, loaded):
    errors = []
    for raw in tokens:
        if any(c.isupper() for c in raw) or raw.count("-") >= 3:
            continue  # noms propres, sigles, débuts de phrase, adresses web
        for part in _split_elisions(raw):
            if part.endswith("'") or part in SPECIAL_WORDS:
                continue
            if any(_is_typo(p, loaded) for p in re.split(r"['-]", part) if p):
                errors.append(("orthographe", raw))
                break
    return errors


def _grammar_errors(seq, words):
    """Règles locales sur les mots d'une phrase (en minuscules, élisions séparées)."""
    errors = []
    for i, w in enumerate(seq):
        prev = seq[i - 1] if i >= 1 else ""
        prev2 = seq[i - 2] if i >= 2 else ""
        nxt = seq[i + 1] if i + 1 < len(seq) else ""
        content = prev not in FUNCTION_WORDS and w not in FUNCTION_WORDS
        # « une crise majeur » : déterminant féminin + nom + adjectif resté au masculin.
        if (content and prev2 in FEMININE_DETERMINERS and prev.endswith("e")
                and _adjective(w, words, feminine=True)):
            errors.append(("accord", f"{prev2} {prev} {w}"))
        # « les problèmes économique » : déterminant pluriel + nom pluriel + adjectif singulier.
        elif (content and prev2 in PLURAL_DETERMINERS and prev[-1:] in ("s", "x")
              and w[-1:] not in ("s", "x", "z") and len(w) > 3 and _adjective(w, words)):
            errors.append(("accord", f"{prev2} {prev} {w}"))
        # « il à été » au lieu de « il a été ».
        if w == "à" and prev in ("il", "elle", "on", "qui") and nxt in AFTER_A:
            errors.append(("confusion", f"{prev} à {nxt}"))
        # « il a manger » : auxiliaire avoir + infinitif au lieu du participe.
        if (prev in AVOIR and prev2 in SUBJECTS and w.endswith("er") and len(w) > 4
                and w not in FUNCTION_WORDS and _known(w[:-2] + "é", words)):
            errors.append(("confusion", f"{prev} {w}"))
        # « il va mangé », « pour mangé » : participe au lieu de l'infinitif.
        if (prev in MODALS and (prev2 in SUBJECTS or prev in ("pour", "sans"))
                and w.endswith("é") and len(w) > 4 and _known(w[:-1] + "er", words)):
            errors.append(("confusion", f"{prev} {w}"))
        # « ils mange » : sujet pluriel + verbe au singulier.
        if (prev in ("ils", "elles") and w.endswith("e") and len(w) > 4
                and w not in FUNCTION_WORDS and _known(w + "nt", words)):
            errors.append(("confusion", f"{prev} {w}"))
    return errors


def find_errors(text):
    """Liste de (type, extrait) des fautes probables du texte."""
    loaded = _dictionary()
    if loaded is None:
        return []
    words = loaded[0]
    text = str(text)
    errors = _spelling_errors(TOKEN_RE.findall(text), loaded)
    for sentence in re.split(r"[.!?;:…\n()«»\"]+", text):
        # Les élisions restent des mots à part (« d ») pour ne pas coller
        # « modèles » et « apprentissage » dans « modèles d'apprentissage ».
        seq = [p.rstrip("'") for t in TOKEN_RE.findall(sentence) for p in _split_elisions(t)]
        errors += _grammar_errors(seq, words)
    return errors


# Ajustement du score : chaque faute divise la « cote » IA (p / (1 - p)).
# Les corpus d'entraînement ne permettent pas d'apprendre ce poids (textes HC3
# traduits automatiquement, fautes rares dans les textes humains publiés) :
# c'est une hypothèse explicite et prudente, mesurée sur les tests.
GRAMMAR_FACTOR = 0.4      # accord ou confusion : rarement fait par un LLM
SPELLING_FACTOR = 0.6     # faute d'orthographe isolée
MAX_SPELLING = 3          # au-delà, accents omis partout = choix délibéré, pas une faute
MIN_FACTOR = 0.1


def human_evidence_factor(errors):
    """Facteur (<= 1) appliqué à la cote IA selon les fautes trouvées."""
    grammar = sum(1 for kind, _ in errors if kind in ("accord", "confusion"))
    spelling = sum(1 for kind, _ in errors if kind == "orthographe")
    if spelling > MAX_SPELLING:
        spelling = 0
    return max(MIN_FACTOR, GRAMMAR_FACTOR ** grammar * SPELLING_FACTOR ** spelling)


def adjust_probability(probability, factor):
    p = min(max(float(probability), 1e-6), 1 - 1e-6)
    odds = p / (1 - p) * factor
    return odds / (1 + odds)
