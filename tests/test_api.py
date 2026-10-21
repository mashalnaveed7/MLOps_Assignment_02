from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "1.0.0"
    assert data["feature_count"] == 30
    assert data["positive_class"] == "malignant"


def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "malignant_predictions" in data
    assert "benign_predictions" in data


def test_predict_with_valid_features():
    import pandas as pd
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "raw" / "breast_cancer.csv")
    features = df.drop(columns=["target"]).iloc[0].to_dict()

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 200
    result = response.json()
    assert result["label"] in ("malignant", "benign")
    assert result["prediction"] in (0, 1)
    assert result["model_version"] == "1.0.0"
    assert 0 <= result["malignant_probability"] <= 1


def test_predict_rejects_missing_features():
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422
