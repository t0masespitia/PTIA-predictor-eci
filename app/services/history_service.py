from app.api.dtos.response.history_item import HistoryItem
from app.api.dtos.response.history_response import HistoryResponse
from app.repository.base_repository import AbstractPredictionRepository


def get_history(repo: AbstractPredictionRepository) -> HistoryResponse:
    records = repo.find_all()
    predictions = [
        HistoryItem(
            id=str(record.id),
            engine_id=record.engine_id,
            rul_predicted=record.predicted_rul,
            timestamp=record.timestamp,
        )
        for record in records
    ]
    return HistoryResponse(count=len(predictions), predictions=predictions)
