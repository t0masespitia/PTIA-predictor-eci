import sys
sys.path.append(".")

from datetime import datetime, timezone

from app.models.entities.prediction_record import PredictionRecord
from app.repository.sqlite_repository import SqlitePredictionRepository


def test_save_returns_id(session_factory):
    repo = SqlitePredictionRepository(session_factory)
    record = PredictionRecord(
        engine_id="engine-1",
        predicted_rul=12.5,
        timestamp=datetime.now(timezone.utc),
        input_sequence_json="[]",
    )
    record_id = repo.save(record)
    assert record_id is not None


def test_find_all_empty(session_factory):
    repo = SqlitePredictionRepository(session_factory)
    assert repo.find_all() == []


def test_find_all_after_save(session_factory):
    repo = SqlitePredictionRepository(session_factory)
    repo.save(
        PredictionRecord(
            engine_id="engine-1",
            predicted_rul=12.5,
            timestamp=datetime.now(timezone.utc),
            input_sequence_json="[]",
        )
    )
    repo.save(
        PredictionRecord(
            engine_id="engine-2",
            predicted_rul=18.0,
            timestamp=datetime.now(timezone.utc),
            input_sequence_json="[]",
        )
    )
    records = repo.find_all()
    assert len(records) == 2
