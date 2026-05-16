from pydantic import BaseModel


class MetricsResponse(BaseModel):
    rmse: float
    mae: float
    units_evaluated: int
