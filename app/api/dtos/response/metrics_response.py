from pydantic import BaseModel


class MetricsResponse(BaseModel):
    rmse: float
    mae: float
    n_units_evaluated: int
