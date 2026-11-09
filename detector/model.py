"""Pipeline NLP entièrement local + calibration + explications."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from .features import PerplexityFeatures, StyleFeatures

FRIENDLY_NAMES = {
    "style:log_nb_mots": "longueur du texte",
    "style:longueur_moy_mot": "longueur moyenne des mots",
    "style:richesse_vocab_mattr": "richesse du vocabulaire",
    "style:taux_hapax": "mots utilisés une seule fois",
    "style:longueur_moy_phrase": "longueur moyenne des phrases",
    "style:ecart_type_phrase": "irrégularité de la longueur des phrases",
    "style:variation_phrases": "variation relative des phrases",
    "style:virgules": "virgules",
    "style:points_virgules": "points-virgules",
    "style:deux_points": "deux-points",
    "style:exclamations": "points d'exclamation",
    "style:interrogations": "points d'interrogation",
    "style:parentheses": "parenthèses",
    "style:guillemets": "guillemets",
    "style:points_suspension": "points de suspension",
    "style:tirets": "tirets",
    "style:chiffres": "chiffres",
    "style:majuscules": "majuscules",
    "style:premiere_personne": "première personne (je, nous...)",
    "style:deuxieme_personne": "deuxième personne (tu, vous...)",
    "style:connecteurs_llm": "connecteurs typiques (cependant, en outre...)",
    "style:hesitations": "hésitations (je pense, peut-être...)",
    "style:familier": "registre familier",
    "style:mots_outils": "mots-outils",
    "style:repetition_bigrammes": "répétitions de groupes de mots",
    "style:debut_phrase_minuscule": "phrases commençant en minuscule",
    "style:espaces_typo_fr": "espaces typographiques françaises",
    "llm:perplexite_log": "prévisibilité pour un modèle de langue",
    "llm:perplexite_log_instruit": "prévisibilité pour un modèle instruit",
    "llm:entropie_croisee": "incertitude attendue du modèle",
    "llm:score_binoculars": "score Binoculars",
    "llm:variabilite_perte": "régularité de la surprise mot à mot",
    "llm:part_tokens_top1": "mots prédits en 1er choix",
}


def build_pipeline(words=True, chars=True, style=True, llm=False, C=1.0, llm_kwargs=None):
    branches = []
    if words:
        branches.append(("mots", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", sublinear_tf=True,
            ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=20_000,
        )))
    if chars:
        branches.append(("caracteres", TfidfVectorizer(
            analyzer="char_wb", lowercase=False, sublinear_tf=True,
            ngram_range=(2, 4), min_df=3, max_features=20_000,
        )))
    if style:
        branches.append(("style", Pipeline([
            ("calcul", StyleFeatures()), ("echelle", StandardScaler()),
        ])))
    if llm:
        branches.append(("llm", Pipeline([
            ("calcul", PerplexityFeatures(**(llm_kwargs or {}))), ("echelle", StandardScaler()),
        ])))
    if not branches:
        raise ValueError("Au moins un type de caractéristiques est nécessaire")
    return Pipeline([
        ("caracteristiques", FeatureUnion(branches)),
        ("classification", LogisticRegression(
            C=C, max_iter=5000, solver="liblinear", random_state=42,
        )),
    ])


class Detector:
    """Pipeline entraîné + calibration (Platt) apprise sur la validation.

    La régression logistique sur TF-IDF donne des scores tassés autour de 0,5 ;
    la calibration les remet à l'échelle pour que le score IA soit interprétable
    et que la zone « indéterminé » ait du sens.
    """

    def __init__(self, pipeline, description=""):
        self.pipeline = pipeline
        self.description = description
        self.calibrator = None

    def fit(self, texts, labels):
        self.pipeline.fit(list(texts), list(labels))
        if list(self.pipeline.named_steps["classification"].classes_) != [0, 1]:
            raise RuntimeError("Ordre des classes inattendu")
        return self

    def decision_function(self, texts):
        return self.pipeline.decision_function(list(texts))

    def calibrate(self, texts, labels):
        scores = self.decision_function(texts).reshape(-1, 1)
        self.calibrator = LogisticRegression(C=1e4).fit(scores, list(labels))
        return self

    def predict_proba(self, texts):
        p = self._calibrate_scores(self.decision_function(texts))
        return np.column_stack([1 - p, p])

    def disable_disk_cache(self):
        """Évite d'écrire sur disque les textes analysés par l'application."""
        union = self.pipeline.named_steps["caracteristiques"]
        for name, branch in union.transformer_list:
            if name == "llm":
                branch.named_steps["calcul"].cache_dir = None
        return self

    def _llm_features(self):
        union = self.pipeline.named_steps["caracteristiques"]
        for name, branch in union.transformer_list:
            if name == "llm":
                return branch.named_steps["calcul"]
        return None

    def record_llm_reference(self):
        """Enregistre les mesures de la phrase de référence (à l'entraînement)."""
        features = self._llm_features()
        self.llm_reference = None if features is None else list(features.reference_values())
        return self

    def check_llm(self, tolerance=1e-3):
        """Vrai si le modèle de langue redonne les valeurs de l'entraînement.

        Protège contre un environnement incohérent (version de torch ou de
        transformers différente, processus dans un état anormal...), qui
        fausserait silencieusement tous les scores.
        """
        features = self._llm_features()
        reference = getattr(self, "llm_reference", None)
        if features is None or reference is None:
            return True
        current = features.reference_values()
        return bool(np.allclose(current, reference, rtol=tolerance, atol=tolerance))

    @property
    def uses_llm(self):
        union = self.pipeline.named_steps["caracteristiques"]
        return any(name == "llm" for name, _ in union.transformer_list)

    def _calibrate_scores(self, scores):
        scores = np.asarray(scores, dtype=float)
        if self.calibrator is None:
            return 1 / (1 + np.exp(-scores))
        return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

    def predict_with_explanation(self, text, top=6):
        """Score IA + explication, en ne calculant les caractéristiques qu'une fois."""
        explanation, decision = self._explain(text, top)
        return float(self._calibrate_scores([decision])[0]), explanation

    def explain(self, text, top=6):
        return self._explain(text, top)[0]

    def _explain(self, text, top=6):
        """Contributions (coefficient x valeur) des caractéristiques lisibles.

        Les n-grammes de caractères sont regroupés en une seule ligne car ils
        ne sont pas interprétables individuellement.
        """
        union = self.pipeline.named_steps["caracteristiques"]
        classifier = self.pipeline.named_steps["classification"]
        coefs = classifier.coef_[0]
        decision = float(classifier.intercept_[0])
        offset = 0
        items = []
        for name, branch in union.transformer_list:
            values = branch.transform([text])
            values = values.toarray()[0] if hasattr(values, "toarray") else np.asarray(values)[0]
            width = len(values)
            contrib = values * coefs[offset:offset + width]
            offset += width
            decision += float(contrib.sum())
            if name == "caracteres":
                items.append(("suites de caractères (ensemble)", float(contrib.sum()), "caracteres"))
                continue
            names = branch.get_feature_names_out()
            for i in np.flatnonzero(contrib):
                raw = str(names[i])
                label = FRIENDLY_NAMES.get(raw, f"« {raw} »" if name == "mots" else raw)
                items.append((label, float(contrib[i]), name))
        items = [it for it in items if abs(it[1]) > 1e-6]
        ai = sorted([it for it in items if it[1] > 0], key=lambda it: -it[1])[:top]
        human = sorted([it for it in items if it[1] < 0], key=lambda it: it[1])[:top]
        fmt = lambda group: [{"indice": n, "poids": round(v, 4), "type": t} for n, v, t in group]
        return {"vers_ia": fmt(ai), "vers_humain": fmt(human)}, decision


def build_model(**kwargs):
    """Compatibilité : renvoie un détecteur non entraîné."""
    return Detector(build_pipeline(**kwargs))
