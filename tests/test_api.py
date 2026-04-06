import sys
sys.path.append(".")

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_WINDOW = [[0.5] * 14 for _ in range(30)]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok():
    r = client.post("/predict", json={"window": VALID_WINDOW})
    assert r.status_code == 200
    body = r.json()
    assert "rul_predicted" in body
    assert body["rul_predicted"] >= 0


def test_predict_wrong_features():
    bad_window = [[0.5] * 10 for _ in range(30)]
    r = client.post("/predict", json={"window": bad_window})
    assert r.status_code == 422


def test_predict_empty_window():
    r = client.post("/predict", json={"window": []})
    assert r.status_code == 422
