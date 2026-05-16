from pydantic import BaseModel, Field, field_validator

from app.api.dtos.request.sensor_reading import SensorReading
from app.core.config import settings


class PredictRequest(BaseModel):
    engine_id: str = Field(..., min_length=1)
    sequence: list[SensorReading] = Field(
        ...,
        description="Ventana temporal de exactamente seq_len ciclos con 17 features nombradas",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence_length(cls, value: list[SensorReading]) -> list[SensorReading]:
        if len(value) != settings.SEQ_LEN:
            raise ValueError(f"sequence debe contener exactamente {settings.SEQ_LEN} ciclos")
        return value
