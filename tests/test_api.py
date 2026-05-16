import sys
sys.path.append(".")

from unittest.mock import patch

from tests.conftest import make_reading


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client, valid_window):
    with patch("app.services.model_service.ModelService.predict", return_value=42.5):
        r = client.post("/predict", json={"engine_id": "engine-1", "sequence": valid_window})
    assert r.status_code == 201
    body = r.json()
    assert body["id"]
    assert body["engine_id"] == "engine-1"
    assert body["rul_predicted"] == 42.5
    assert body["timestamp"]
    assert body["unit"] == "cycles"


def test_predict_missing_engine_id(client, valid_window):
    r = client.post("/predict", json={"sequence": valid_window})
    assert r.status_code == 422


def test_predict_missing_field(client):
    bad_reading = make_reading()
    bad_reading.pop("mach_number")
    bad_window = [bad_reading] + [make_reading() for _ in range(29)]
    r = client.post("/predict", json={"engine_id": "engine-1", "sequence": bad_window})
    assert r.status_code == 422


def test_predict_wrong_length(client):
    r = client.post(
        "/predict",
        json={"engine_id": "engine-1", "sequence": [make_reading() for _ in range(29)]},
    )
    assert r.status_code == 422
