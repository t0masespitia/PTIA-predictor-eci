from pydantic import BaseModel, Field
from typing import Optional


class TrainRequest(BaseModel):
    epochs: Optional[int] = Field(None, ge=1, le=500, description="Epocas de entrenamiento")
    seq_len: Optional[int] = Field(None, ge=10, le=100, description="Longitud de ventana")
    batch_size: Optional[int] = Field(None, ge=8, le=256, description="Batch size")


class TrainResponse(BaseModel):
    epochs_run: int
    final_train_loss: float
    final_val_loss: float
    best_model_path: str


class PredictRequest(BaseModel):
    window: list[list[float]] = Field(
        ...,
        description="Ventana temporal: lista de seq_len pasos x 14 sensores"
    )


class PredictResponse(BaseModel):
    rul_predicted: float
    unit: str = "cycles"


class MetricsResponse(BaseModel):
    rmse: float
    mae: float
    n_units_evaluated: int
