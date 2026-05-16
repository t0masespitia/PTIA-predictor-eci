from datetime import datetime

from pydantic import BaseModel


class HistoryItem(BaseModel):
    id: str
    engine_id: str
    rul_predicted: float
    timestamp: datetime
