from fastapi import APIRouter, HTTPException
from app.api.schemas import PredictRequest, PredictResponse
from app.services.prediction_service import predict
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse, summary="Predecir RUL")
def predict_rul(request: PredictRequest):
    seq_len   = len(request.window)
    n_features = len(request.window[0]) if request.window else 0

    if seq_len < 1:
        raise HTTPException(status_code=422, detail="La ventana no puede estar vacia")
    if n_features != 14:
        raise HTTPException(
            status_code=422,
            detail=f"Se esperan 14 features por paso, recibidos: {n_features}"
        )

    try:
        rul = predict(request.window)
        return PredictResponse(rul_predicted=rul)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error en prediccion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
