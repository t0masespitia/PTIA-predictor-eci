from fastapi import APIRouter, HTTPException
from app.api.schemas import TrainRequest, TrainResponse
from app.services.training_service import run_training
from app.services.prediction_service import reset_model
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/train", response_model=TrainResponse, summary="Entrenar modelo CNN-BiLSTM")
def train(request: TrainRequest = TrainRequest()):
    try:
        result = run_training(
            epochs=request.epochs,
            seq_len=request.seq_len,
            batch_size=request.batch_size,
        )
        reset_model()
        return result
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))
