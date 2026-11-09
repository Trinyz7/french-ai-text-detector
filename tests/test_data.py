import pytest

from detector.download_data import normalize_label, paired_sample
from detector.features import STYLE_FEATURE_NAMES, style_features
from detector.inference import text_statistics


@pytest.mark.parametrize("raw, expected", [
    (0, 0), (1, 1), ("HUMAN", 0), ("CHATGPT", 1), ("True", 0), ("False", 1), (True, 0), (False, 1),
])
def test_labels(raw, expected):
    assert normalize_label(raw) == expected


def test_unknown_label():
    with pytest.raises(ValueError):
        normalize_label("other")


def test_statistics():
    stats = text_statistics("Bonjour ! Voici un texte français.")
    assert stats["mots"] >= 5
    assert stats["phrases_approx"] == 2


def _answer(prefix, i):
    return f"{prefix} réponse {i} " + " ".join(f"mot{j}" for j in range(40))


def test_paired_sample_keeps_one_human_and_one_ai_per_question():
    import random

    records = []
    for i in range(10):
        records.append({"id": i, "source": "reddit", "is_human": "True", "answer": _answer("h", i)})
        records.append({"id": i, "source": "reddit", "is_human": "False", "answer": _answer("ia", i)})
    records.append({"id": 99, "source": "reddit", "is_human": "True", "answer": _answer("seule", 99)})
    rows = paired_sample(records, per_source=5, rng=random.Random(0), corpus="test")
    assert len(rows) == 10
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], set()).add(row["label"])
    assert all(labels == {0, 1} for labels in by_group.values())
    assert "reddit:99" not in by_group


def test_style_features_detect_connectors_and_informal_register():
    names = STYLE_FEATURE_NAMES
    formal = dict(zip(names, style_features(
        "Cependant, il est important de noter que plusieurs facteurs entrent en jeu. En outre, tout évolue.")))
    casual = dict(zip(names, style_features(
        "Bah perso j'ai testé ça l'an dernier... franchement c'était nul !! Mon pote pareil.")))
    assert formal["connecteurs_llm"] > casual["connecteurs_llm"]
    assert casual["familier"] > formal["familier"]
    assert casual["premiere_personne"] > formal["premiere_personne"]


def test_strip_markdown_and_prompt():
    from detector.generate_data import build_prompt, strip_markdown

    assert strip_markdown("## Titre\n**Gras** et\n- point") == "Titre\nGras et\npoint"
    prompt = build_prompt({"question": "Pourquoi le ciel est-il bleu ?"}, "linkedin", 120)
    assert "Pourquoi le ciel est-il bleu ?" in prompt and "120 mots" in prompt
    review = build_prompt({"polarity": "négative"}, "neutre", 80)
    assert "critique de film négative" in review
