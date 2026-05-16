import json
from datetime import timezone, datetime

from app.api.dtos.request.predict_request import PredictRequest
from app.api.dtos.response.predict_response import PredictResponse
from app.models.entities.prediction_record import PredictionRecord
from app.repository.base_repository import AbstractPredictionRepository
from app.services.model_service import ModelService


def predict_and_store(
    request: PredictRequest,
    model_service: ModelService,
    repo: AbstractPredictionRepository,
) -> PredictResponse:
    rul_predicted = model_service.predict(request.sequence)
    timestamp = datetime.now(timezone.utc)
    record = PredictionRecord(
        engine_id=request.engine_id,
        predicted_rul=rul_predicted,
        timestamp=timestamp,
        input_sequence_json=json.dumps([reading.model_dump() for reading in request.sequence]),
    )
    record_id = repo.save(record)
    return PredictResponse(
        id=str(record_id),
        engine_id=request.engine_id,
        rul_predicted=rul_predicted,
        timestamp=timestamp,
    )
