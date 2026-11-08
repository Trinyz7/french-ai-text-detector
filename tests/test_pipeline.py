import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from detector.inference import analyze, load_detector
from detector.train import choose_thresholds, train


def make_partition(split, per_class):
    rows = []
    for i in range(per_class):
        rows.append({
            "text": f"{split} témoignage personnel numéro {i}. Hier, j'ai pris le bus en retard "
                    f"et mon ami m'a raconté son voyage au marché. L'histoire concerne le mardi {i}.",
            "label": 0,
        })
        rows.append({
            "text": f"{split} explication structurée numéro {i}. En premier lieu, il convient "
                    f"de considérer la question dans un cadre méthodologique. Pour conclure, l'analyse {i}.",
            "label": 1,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def trained(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    for split, count in (("train", 24), ("validation", 10), ("test", 10)):
        make_partition(split, count).to_csv(directory / f"{split}.csv", index=False)
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "metrics.json"
    train(directory, model_path, meta_path)
    model, metadata = load_detector(model_path, meta_path)
    return model, metadata, model_path, meta_path


def test_thresholds():
    assert choose_thresholds([0] * 10 + [1] * 10, [0.1] * 10 + [0.9] * 10)["ai"] >= 0.5


def test_train_and_predict(trained):
    model, meta, model_path, meta_path = trained
    assert model_path.is_file() and meta_path.is_file()
    assert len(json.loads(meta_path.read_text())["test_seuil_0_5"]["matrice_confusion"]) == 2
    out = analyze("Je suis allé dans le centre-ville et j'ai oublié mon sac de sport.", model, meta)
    assert 0 <= out["score_ia"] <= 1
    assert out["categorie"] in {"indices_humains", "indices_ia", "indetermine"}
    assert any("court" in note for note in out["avertissements"])


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
