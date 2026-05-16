from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dtos import PredictRequest, PredictResponse
from app.core.logging import get_logger
from app.repository.base_repository import AbstractPredictionRepository
from app.repository.sqlite_repository import SqlitePredictionRepository
from app.services.model_service import ModelService
from app.services.prediction_service import predict_and_store

logger = get_logger(__name__)
router = APIRouter()


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_repo(request: Request) -> AbstractPredictionRepository:
    return SqlitePredictionRepository(request.app.state.session_factory)


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Predecir RUL",
)
def predict_rul(
    body: PredictRequest,
    model: ModelService = Depends(get_model_service),
    repo: AbstractPredictionRepository = Depends(get_repo),
):
    try:
        return predict_and_store(body, model, repo)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error en prediccion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
