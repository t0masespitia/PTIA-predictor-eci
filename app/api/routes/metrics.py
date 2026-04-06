from fastapi import APIRouter, HTTPException
from app.api.schemas import MetricsResponse
from app.services.evaluation_service import compute_metrics
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse, summary="Evaluar modelo sobre test set")
def get_metrics():
    try:
        return compute_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error en evaluacion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
