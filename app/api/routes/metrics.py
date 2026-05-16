from fastapi import APIRouter, Depends, HTTPException

from app.api.dtos import MetricsResponse
from app.services.evaluation_service import compute_metrics
from app.core.logging import get_logger
from app.services.model_service import ModelService
from app.api.routes.prediction import get_model_service

logger = get_logger(__name__)
router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse, summary="Evaluar modelo sobre test set")
def get_metrics(model: ModelService = Depends(get_model_service)):
    try:
        return compute_metrics(model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error en evaluacion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
