import sys
sys.path.append(".")

from unittest.mock import patch


def test_history_empty(client):
    r = client.get("/predictions")
    assert r.status_code == 200
    assert r.json() == {"count": 0, "predictions": []}


def test_history_after_predict(client, valid_window):
    with patch("app.services.model_service.ModelService.predict", return_value=17.5):
        predict_response = client.post(
            "/predict",
            json={"engine_id": "engine-1", "sequence": valid_window},
        )

    assert predict_response.status_code == 201

    history_response = client.get("/predictions")
    body = history_response.json()
    assert history_response.status_code == 200
    assert body["count"] == 1
    assert body["predictions"][0]["engine_id"] == "engine-1"


def test_history_multiple_engines(client, valid_window):
    with patch("app.services.model_service.ModelService.predict", return_value=33.3):
        for engine_id in ["engine-1", "engine-2", "engine-3"]:
            response = client.post(
                "/predict",
                json={"engine_id": engine_id, "sequence": valid_window},
            )
            assert response.status_code == 201

    history_response = client.get("/predictions")
    body = history_response.json()
    assert body["count"] == 3
    assert [item["engine_id"] for item in body["predictions"]] == [
        "engine-1",
        "engine-2",
        "engine-3",
    ]
