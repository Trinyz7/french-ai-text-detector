import json
import pickle

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from detector.features import PerplexityFeatures
from detector.inference import analyze, load_detector
from detector.train import choose_thresholds, train


def make_partition(split, per_class, sources=("reddit", "finance", "wiki")):
    rows = []
    for i in range(per_class):
        source = sources[i % len(sources)]
        rows.append({
            "text": f"{split} témoignage personnel numéro {i}. Hier, j'ai pris le bus en retard "
                    f"et mon ami m'a raconté son voyage au marché. Bref, c'était le mardi {i} !",
            "label": 0, "source": source,
        })
        rows.append({
            "text": f"{split} explication structurée numéro {i}. Cependant, il convient "
                    f"de considérer la question dans un cadre méthodologique. En outre, l'analyse {i}.",
            "label": 1, "source": source,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def trained(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    for split, count in (("train", 30), ("validation", 24), ("test", 24)):
        make_partition(split, count).to_csv(directory / f"{split}.csv", index=False)
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "metrics.json"
    train(directory, model_path, meta_path, config_name="sans_llm", verbose=False)
    model, metadata = load_detector(model_path, meta_path)
    return model, metadata, model_path, meta_path


def test_thresholds_keep_uncertain_band():
    thresholds = choose_thresholds([0] * 10 + [1] * 10, [0.1] * 10 + [0.9] * 10)
    assert thresholds["ai"] >= 0.65
    assert thresholds["human"] <= 0.35


def test_train_and_predict(trained):
    model, meta, model_path, meta_path = trained
    assert model_path.is_file() and meta_path.is_file()
    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(saved["test"]["seuil_0_5"]["matrice_confusion"]) == 2
    assert "par_source" in saved["test"]
    assert "moyenne_auc" in saved["hors_domaine"]
    out = analyze("Je suis allé dans le centre-ville et j'ai oublié mon sac de sport.", model, meta)
    assert 0 <= out["score_ia"] <= 1
    assert out["categorie"] in {"indices_humains", "indices_ia", "indetermine"}
    assert any("court" in note for note in out["avertissements"])
    assert set(out["explication"]) == {"vers_ia", "vers_humain"}


def test_explanation_matches_score(trained):
    model, *_ = trained
    text = "Cependant, il convient de noter que la question est complexe. En outre, plusieurs facteurs."
    score, _ = model.predict_with_explanation(text)
    assert score == pytest.approx(model.predict_proba([text])[0, 1], abs=1e-6)


def test_perplexity_features_are_not_pickled():
    feats = PerplexityFeatures(cache_dir=None)
    feats._models = ("poids", "lourds")
    assert "_models" not in pickle.loads(pickle.dumps(feats)).__dict__


def test_fastapi(trained, monkeypatch):
    import api

    model, meta, _, _ = trained
    monkeypatch.setattr(api, "get_components", lambda: (model, meta))
    client = TestClient(api.app)
    assert client.get("/health").json()["model_loaded"]
    response = client.post("/predict", json={"text": "Bonjour. Voici une petite phrase pour le test."})
    assert response.status_code == 200
    assert "score_ia" in response.json()
    assert len(client.post("/predict/batch", json={"texts": ["Un", "Deux"]}).json()["results"]) == 2
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_llm_reference_check_detects_drift(trained, monkeypatch):
    model, meta, *_ = trained

    class FakeLM:
        values = [1.0, 2.0, 3.0]

        def reference_values(self):
            return self.values

    fake = FakeLM()
    monkeypatch.setattr(model, "_llm_features", lambda: fake)
    model.record_llm_reference()
    assert model.check_llm()
    fake.values = [1.0, 2.5, 3.0]
    assert not model.check_llm()
    with pytest.raises(RuntimeError):
        analyze("Un texte quelconque pour vérifier le contrôle de cohérence.", model, meta)
