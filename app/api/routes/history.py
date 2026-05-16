from fastapi import APIRouter, Depends, HTTPException

from app.api.dtos import HistoryResponse
from app.api.routes.prediction import get_repo
from app.core.logging import get_logger
from app.repository.base_repository import AbstractPredictionRepository
from app.services.history_service import get_history

logger = get_logger(__name__)
router = APIRouter()


@router.get("/predictions", response_model=HistoryResponse, summary="Consultar historico")
def list_predictions(repo: AbstractPredictionRepository = Depends(get_repo)):
    try:
        return get_history(repo)
    except Exception as e:
        logger.error(f"Error consultando historico: {e}")
        raise HTTPException(status_code=500, detail=str(e))
