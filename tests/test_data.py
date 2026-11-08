import pytest

from detector.download_data import normalize_label
from detector.inference import text_statistics


@pytest.mark.parametrize("raw, expected", [(0, 0), (1, 1), ("HUMAN", 0), ("CHATGPT", 1)])
def test_labels(raw, expected):
    assert normalize_label(raw) == expected


def test_unknown_label():
    with pytest.raises(ValueError):
        normalize_label("other")


def test_statistics():
    stats = text_statistics("Bonjour ! Voici un texte français.")
    assert stats["mots"] >= 5
    assert stats["phrases_approx"] == 2
