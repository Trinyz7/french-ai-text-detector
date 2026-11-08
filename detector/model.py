"""Pipeline NLP explicite et entièrement local."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_model():
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("mots", TfidfVectorizer(
                lowercase=True, strip_accents="unicode", sublinear_tf=True,
                ngram_range=(1, 2), min_df=2, max_df=0.99,
                max_features=35_000,
            )),
            ("caracteres", TfidfVectorizer(
                analyzer="char_wb", lowercase=True, strip_accents="unicode",
                sublinear_tf=True, ngram_range=(3, 5), min_df=2,
                max_features=35_000,
            )),
        ])),
        ("classification", LogisticRegression(
            max_iter=1200, class_weight="balanced", solver="liblinear",
            random_state=42,
        )),
    ])
