import pandas as pd
import pytest

from detector.errors import adjust_probability, find_errors, human_evidence_factor
from detector.train import short_excerpts

pytest.importorskip("spellchecker")


def kinds(text):
    return sorted(kind for kind, _ in find_errors(text))


def test_detects_typical_human_errors():
    errors = dict((excerpt, kind) for kind, excerpt in find_errors(
        "Il à été malade. Ils mange des choses simple. Il a manger hier, "
        "c'est egalement une crise majeur."))
    assert errors["egalement"] == "orthographe"
    assert errors["il à été"] == "confusion"
    assert errors["ils mange"] == "confusion"
    assert errors["a manger"] == "confusion"
    assert errors["des choses simple"] == "accord"
    assert errors["une crise majeur"] == "accord"


@pytest.mark.parametrize("text", [
    "La voiture rouge est garée dans la rue. Une fois par an, les enfants jouent ensemble.",
    "Les gens bien sont rares. Des modèles d'apprentissage. La pratique consistant à manger.",
    "Les personnes vivant en France utilisent des pauses café et des processus métier.",
    "Aujourd'hui, j'ai entrainé mon chien ; les photons et les quarks sont des particules.",
    "Le cœur de l'œuvre est une question économique majeure pour les marchés européens.",
])
def test_correct_french_is_not_flagged(text):
    assert find_errors(text) == []


def test_errors_lower_the_ai_score():
    assert human_evidence_factor([]) == 1
    one = human_evidence_factor(find_errors("Une crise majeur."))
    assert one < 1
    assert adjust_probability(0.8, one) < 0.8
    assert adjust_probability(0.8, 1) == pytest.approx(0.8)


def test_systematic_missing_accents_do_not_count_as_human():
    # Texte où tous les accents manquent : choix délibéré (Gemini le fait).
    errors = find_errors("Il etait une fois un systeme tres elabore, deja celebre a l'epoque.")
    assert sum(kind == "orthographe" for kind, _ in errors) > 3
    assert human_evidence_factor(errors) == 1


def test_short_excerpts_are_short_and_keep_labels():
    long_text = " ".join(f"Phrase numéro {i} avec quelques mots de plus pour la longueur." for i in range(30))
    frame = pd.DataFrame({"text": [long_text] * 3, "label": [0, 1, 0], "source": ["a", "b", "c"]})
    frame["text"] = [t + f" Fin {i}." for i, t in enumerate(frame["text"])]
    out = short_excerpts(frame)
    assert len(out) == 3
    assert out["text"].str.split().str.len().between(25, 75).all()
    assert list(out["label"]) == [0, 1, 0]
